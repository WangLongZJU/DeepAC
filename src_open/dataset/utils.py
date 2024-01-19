import pickle
import numpy as np
import os.path as osp

import torch
import cv2
import imgaug.augmenters as iaa

def read_template_data(obj_names, paths):
    num_sample_contour_points = {}
    template_views = {}
    orientations = {}
    for obj_name, path in zip(obj_names, paths):
        with open(path, "rb") as pkl_handle:
            pre_render_dict = pickle.load(pkl_handle)
        head = pre_render_dict['head']
        num_sample_contour_points[obj_name] = head['num_sample_contour_point']
        template_views[obj_name] = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
        orientations[obj_name] = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)

    return num_sample_contour_points, template_views, orientations

def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image

def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale

def crop(image, bbox2d, camera=None, return_bbox=False):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    half_w_new, half_h_new = bbox2d[2:].astype(np.int) // 2
    x, y = bbox2d[:2].astype(np.int)
    left = np.clip(x - half_w_new, 0, w - 1)
    right = np.clip(x + half_w_new, 0, w - 1)
    top = np.clip(y - half_h_new, 0, h - 1)
    bottom = np.clip(y + half_h_new, 0, h - 1)

    image = image[top:bottom, left:right]
    ret = [image]
    if camera is not None:
        ret += [camera.crop((left, top), (half_w_new*2, half_h_new*2))]
    if return_bbox:
        ret += [(top, bottom, left, right)]
    return ret

def zero_pad(size, *images):
    ret = []
    for image in images:
        h, w = image.shape[:2]
        padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret

def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()

def get_imgaug_seq():
    seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
    ], random_order=True)  # apply augmenters in random order

    return seq
