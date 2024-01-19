import json
import os
from pathlib import Path
import glob
import cv2

from pytorch3d.ops import sample_farthest_points
from .base_dataset import BaseDataset, set_seed
import torch
from torch.utils.data import ConcatDataset
import numpy as np
from ..utils.geometry.wrappers import Pose, Camera
from ..utils.utils import project_correspondences_line, get_closest_template_view_index, \
    get_closest_k_template_view_index, generate_random_aa_and_t, get_bbox_from_p2d

from .utils import read_template_data, read_image, resize, numpy_image_to_torch, crop, zero_pad, get_imgaug_seq
import logging
from tqdm import tqdm
from ..utils.draw_tutorial import draw_correspondence_lines_in_image
import imgaug as ia
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
from pytorch3d.structures import Meshes

from .HOPE import _Dataset_HOPE
from .ICBIN import _Dataset_ICBIN
from .ICMI import _Dataset_ICMI
from .TLESS import _Dataset_TLESS
from .TUDL import _Dataset_TUDL
from .LM import _Dataset_LM
from .YCBV import _Dataset_YCBV
from .RUAPC import _Dataset_RUAPC


logger = logging.getLogger(__name__)


class BOP(BaseDataset):
    default_conf = {
        'dataset_dir': '',
        'background_image_dir': '',
        
        'train_num_per_obj': 1500,
        'val_num_per_obj': 500,
        'random_sample': True,

        'get_top_k_template_views': 1,
        'skip_template_view': 1,
        'geometry_unit_in_meter': 0.001,  # must equal to the geometry_unit_in_meter of preprocess
        'offset_angle_step': 5.0,
        'min_offset_angle': 5.0,  # 5.0,
        'max_offset_angle': 15.0,  # 25.0,
        'offset_translation_step': 0.01,
        'min_offset_translation': 0.005,  # 0.01,# 0.01,  # meter
        'max_offset_translation': 0.015,  # 0.025,# 0.03,  # meter
        'val_offset': True,
        'train_offset': True,
        'skip_frame': 1,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': False,
        'crop_border': None,
        'pad': None,
        'change_background': False,
        'change_background_thres': 0.5,
        'img_aug': False,
        'seed': 0,
        'sample_vertex_num': 500,

        'opt': False,
        'debug_check_display': False
    }

    strict_conf = False

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        # assert split != 'test', 'Not supported'
        # if split == 'train':
        #     return ConcatDataset([_Dataset_ICBIN(self.conf, split),
        #                           _Dataset_TLESS(self.conf, split),
        #                           _Dataset_TUDL(self.conf, split),
        #                           _Dataset_LM(self.conf, split),
        #                           _Dataset_YCBV(self.conf, split),])
        # elif split == 'val':
        #     return ConcatDataset([_Dataset_HOPE(self.conf, split),
        #                           _Dataset_ICBIN(self.conf, split),
        #                           _Dataset_ICMI(self.conf, split),
        #                           _Dataset_TLESS(self.conf, split),
        #                           _Dataset_TUDL(self.conf, split),
        #                           _Dataset_LM(self.conf, split),
        #                           _Dataset_YCBV(self.conf, split),]) 
        # if split == 'train':
        #     return ConcatDataset([_Dataset_RUAPC(self.conf, split)])
        # elif split == 'val':
        #     return ConcatDataset([_Dataset_RUAPC(self.conf, split)]) 
        if split == 'train' or split == 'val':
            return _Dataset(self.conf, split)
        elif split == 'test':
            # TODO: implement later
            return _Dataset_test(self.conf, split)
        else:
            raise NotImplementedError        


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(conf.dataset_dir)
            
        self.conf, self.split = conf, split

        self.geometry_unit_in_meter = float(conf.geometry_unit_in_meter)
        self.min_offset_angle = float(conf.min_offset_angle)
        self.max_offset_angle = float(conf.max_offset_angle)
        self.min_offset_translation = float(conf.min_offset_translation)
        self.max_offset_translation = float(conf.max_offset_translation)

        # check background if change_background is True
        if conf.change_background is True:
            self.background_image_dir = Path(conf.background_image_dir, 'JPEGImages')
            assert self.background_image_dir.exists()
            self.background_image_path = np.stack(os.listdir(str(self.background_image_dir)))

        # TODO: add RUAPC dataset
        self.dataset_list = []

        for sub_dataset_name in conf.sub_dataset_dir:
            if conf[sub_dataset_name+'_'+split+'_obj_names'] != 'none':
                if sub_dataset_name == 'hope':
                    self.dataset_list.append(_Dataset_HOPE(self.conf, split))
                elif sub_dataset_name == 'icbin':
                    self.dataset_list.append(_Dataset_ICBIN(self.conf, split))
                elif sub_dataset_name == 'icmi':
                    self.dataset_list.append(_Dataset_ICMI(self.conf, split))
                elif sub_dataset_name == 'tless':
                    self.dataset_list.append(_Dataset_TLESS(self.conf, split))
                elif sub_dataset_name == 'tudl':
                    self.dataset_list.append(_Dataset_TUDL(self.conf, split))
                elif sub_dataset_name == 'lm':
                    self.dataset_list.append(_Dataset_LM(self.conf, split))
                elif sub_dataset_name == 'ycbv':
                    self.dataset_list.append(_Dataset_YCBV(self.conf, split))
                elif sub_dataset_name == 'ruapc':
                    pass # not implement
                else:
                    raise NotImplementedError 

        # if split == 'train':
        #     self.dataset_list = [_Dataset_ICBIN(self.conf, split),
        #                         _Dataset_TLESS(self.conf, split),
        #                         _Dataset_TUDL(self.conf, split),
        #                         _Dataset_LM(self.conf, split),
        #                         _Dataset_YCBV(self.conf, split)]
        # elif split == 'val':
        #     self.dataset_list = [_Dataset_HOPE(self.conf, split),
        #                         _Dataset_ICBIN(self.conf, split),
        #                         _Dataset_ICMI(self.conf, split),
        #                         _Dataset_TLESS(self.conf, split),
        #                         _Dataset_TUDL(self.conf, split),
        #                         _Dataset_LM(self.conf, split),
        #                         _Dataset_YCBV(self.conf, split)] 
        # else:
        #     raise NotImplementedError 

        if split != 'test':
            self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images with seed {seed}')
        set_seed(seed)

        self.items = []
        for dataset in self.dataset_list:
            dataset.sample_new_items(seed)
            self.items.extend(dataset.items)

        if self.conf.change_background is True:
            selected = np.random.RandomState(seed).choice(
                len(self.background_image_path), len(self.items), replace=True)
            self.selected_background_image_path = self.background_image_path[selected]
        if self.conf.img_aug:
            ia.seed(seed)

    def update_offset_angle_and_translation(self):
        # if self.max_offset_angle < 45:
        #     self.min_offset_angle += self.conf.offset_angle_step
        #     self.max_offset_angle += self.conf.offset_angle_step
        # else:
        #     self.min_offset_angle = 5
        #     self.max_offset_angle = 45
        # if self.max_offset_translation < 0.06:
        #     self.min_offset_translation += self.conf.offset_translation_step
        #     self.max_offset_translation += self.conf.offset_translation_step
        # else:
        #     self.min_offset_translation = 0.01
        #     self.max_offset_translation = 0.06

        logger.info(f'Offset angle: {self.min_offset_angle}, {self.max_offset_angle}')
        logger.info(f'Offset translation: {self.min_offset_translation}, {self.max_offset_translation}')

    def image_aug(self, img):
        seq = get_imgaug_seq()
        img_aug = seq(image=img)
        return img_aug

    def read_image(self, image_path, conf, camera: Camera, bbox2d, image=None, img_aug=False):

        # read image
        if image is None:
            img = read_image(image_path, conf.grayscale)
        else:
            img = image

        if conf.crop:
            if conf.crop_border:
                bbox2d[2:] += conf.crop_border * 2
            img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)

        if conf.resize:
            scales = (1, 1)
            if isinstance(conf.resize, int):
                if conf.resize_by == 'max':
                    # print('img shape', img.shape)
                    # print('img path', image_path)
                    img, scales = resize(img, conf.resize, fn=max)
                elif (conf.resize_by == 'min' or (conf.resize_by == 'min_if' and min(*img.shape[:2]) < conf.resize)):
                    img, scales = resize(img, conf.resize, fn=min)
            elif len(conf.resize) == 2:
                img, scales = resize(img, list(conf.resize))

            if scales != (1, 1):
                camera = camera.scale(scales)

        # if conf.change_background:
        #     raise NotImplementedError

        if conf.pad:
            img, = zero_pad(conf.pad, img)

        if img_aug:
            img_aug = self.image_aug(img)
        else:
            img_aug = img
        img_aug = img_aug.astype(np.float32)
        img = img.astype(np.float32)

        return numpy_image_to_torch(img), numpy_image_to_torch(img_aug), camera

    def transform_img(self, img, bbox2d, conf):
        if conf.crop:
            if conf.crop_border:
                bbox2d[2:] += conf.crop_border * 2
            img, bbox = crop(img, bbox2d, camera=None, return_bbox=True)

        if conf.resize:
            if isinstance(conf.resize, int):
                if conf.resize_by == 'max':
                    # print('img shape', img.shape)
                    # print('img path', image_path)
                    img, _ = resize(img, conf.resize, fn=max)
                elif (conf.resize_by == 'min' or (conf.resize_by == 'min_if' and min(*img.shape[:2]) < conf.resize)):
                    img, _ = resize(img, conf.resize, fn=min)
            elif len(conf.resize) == 2:
                img, _ = resize(img, list(conf.resize))

        if conf.pad:
            img, = zero_pad(conf.pad, img)

        return numpy_image_to_torch(img)

    def read_mask(self, mask_path, mask_visib_path, bbox2d, conf):
        mask = read_image(mask_path, True)
        mask_visib = read_image(mask_visib_path, True)

        mask_edge = cv2.Canny(mask, 100, 200)
        mask_visib_edge = cv2.Canny(mask_visib, 100, 200)

        # edge_visib = mask_visib
        edge_visib = mask_edge & mask_visib_edge
        # edge_non_visib = mask_edge ^ edge_visib

        return self.transform_img(edge_visib, bbox2d.copy(), conf), self.transform_img(mask_edge, bbox2d.copy(), conf), \
               self.transform_img(mask_visib, bbox2d.copy(), conf)
        # if conf.crop:
        #     if conf.crop_border:
        #         bbox2d[2:] += conf.crop_border * 2
        #     edge_visib, bbox = crop(edge_visib, bbox2d, camera=None, return_bbox=True)
        #
        # if conf.resize:
        #     if isinstance(conf.resize, int):
        #         if conf.resize_by == 'max':
        #             # print('img shape', img.shape)
        #             # print('img path', image_path)
        #             edge_visib, _ = resize(edge_visib, conf.resize, fn=max)
        #         elif (conf.resize_by == 'min' or (conf.resize_by == 'min_if' and min(*edge_visib.shape[:2]) < conf.resize)):
        #             edge_visib, _ = resize(edge_visib, conf.resize, fn=min)
        #     elif len(conf.resize) == 2:
        #         edge_visib, _ = resize(edge_visib, list(conf.resize))
        #
        # if conf.pad:
        #     edge_visib, = zero_pad(conf.pad, edge_visib)
        #
        # return numpy_image_to_torch(edge_visib)  # torch.from_numpy(mask_edge[None])

    def draw_mask(self, template_views, gt_body2view_pose, orientations_in_body, n_sample, camera, image):
        gt_index = get_closest_template_view_index(gt_body2view_pose, orientations_in_body)
        gt_template_view = template_views[gt_index * n_sample:(gt_index + 1) * n_sample, :]
        data_lines = project_correspondences_line(gt_template_view, gt_body2view_pose, camera)
        gt_centers_in_image = data_lines['centers_in_image'].unsqueeze(1).numpy().astype(np.int)
        mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = cv2.drawContours(mask, [gt_centers_in_image], -1, 1, -1)

        return mask

    def change_background(self, idx, image, mask, thres):
        
        if np.random.rand() > thres:
            return image

        background_path = Path(self.background_image_dir, self.selected_background_image_path[idx])
        background_image = read_image(background_path, self.conf.grayscale)
        background_image, _ = resize(background_image, image.shape[:2])
        mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        img = np.where(mask == 0, background_image, image)
        # img = torch.where(mask.expand(3, -1, -1) == 0, background_image, image)
        # img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./test.png', img)

        return img

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item['image_path']
        img_id = item['img_id']
        output_name = item['output_name']
        ori_image = read_image(image_path, self.conf.grayscale)
        obj_id = item['obj_id']
        body2view_R = item['body2view_R'].reshape(3, 3)
        body2view_t = item['body2view_t']
        gt_body2view_pose = Pose.from_Rt(body2view_R, body2view_t)
        K = item['K']
        intrinsic_param = torch.tensor([ori_image.shape[1], ori_image.shape[0],
                                        K[0], K[4], K[2], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)
        orientations_in_body = item['orientations_in_body']
        template_views = item['template_views']
        n_sample = item['n_sample']
        diameter = item['diameter']


        # generate offset to ground truth pose
        if (img_id == 0) or (self.split == 'train' and self.conf.train_offset) or \
                (self.split == 'val' and self.conf.val_offset):  # self.split == 'train' or self.conf.val_offset:
            random_aa, random_t = generate_random_aa_and_t(self.min_offset_angle, self.max_offset_angle,
                                                           self.min_offset_translation, self.max_offset_translation)
            random_pose = Pose.from_aa(random_aa, random_t)
            body2view_pose = gt_body2view_pose @ random_pose[0]
        else:
            # last_body2view_R = item['last_body2view_R'].reshape(3, 3)
            # last_body2view_t = item['last_body2view_t']
            # body2view_pose = Pose.from_Rt(last_body2view_R, last_body2view_t)
            raise NotImplementedError

        # get closest template view
        indices = get_closest_k_template_view_index(body2view_pose,
                                                    orientations_in_body,
                                                    self.conf.get_top_k_template_views * self.conf.skip_template_view)
        closest_template_views = torch.stack([template_views[ind * n_sample:(ind + 1) * n_sample, :]
                                              for ind in indices[::self.conf.skip_template_view]])
        closest_orientations_in_body = orientations_in_body[indices[::self.conf.skip_template_view]]

        # calc bbox
        data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])

        # read image
        image, aug_image, camera = self.read_image(image_path, self.conf, ori_camera, bbox2d.numpy().copy(), ori_image,
                                                   img_aug=self.conf.img_aug if self.split == 'train' else False)

        if self.conf.change_background and self.split == 'train':
            ori_mask = self.draw_mask(template_views, gt_body2view_pose, orientations_in_body,
                                      n_sample, ori_camera, ori_image)
            ori_image_with_background = self.change_background(idx, ori_image, ori_mask, self.conf.change_background_thres)
            if self.conf.img_aug == True and self.split == 'train':
                aug_image_with_background = self.image_aug(ori_image_with_background)
                aug_image = self.transform_img(aug_image_with_background, bbox2d.numpy().copy(), self.conf)
            image = self.transform_img(ori_image_with_background, bbox2d.numpy().copy(), self.conf)

        # aug_image = (aug_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./aug_img.png', aug_image)
        # image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./img.png', image)
        # import ipdb
        # ipdb.set_trace()

        # read mask
        # mask_path = item['mask_path']
        # mask_visib_path = item['mask_visib_path']
        # edge_visib, edge, mask_visib = self.read_mask(mask_path, mask_visib_path, bbox2d.numpy().copy(), self.conf)

        # if self.conf.change_background:
        #     image = self.change_background(idx, image, mask_visib)

        # check
        if self.conf.debug_check_display:
            data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, camera)
            display_image = draw_correspondence_lines_in_image((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                                               data_lines['centers_in_image'],
                                                               data_lines['centers_valid'],
                                                               data_lines['normals_in_image'], 10)
            display_path = Path(os.path.basename(image_path))
            cv2.imwrite(str(display_path), display_image)

            # display_path = Path(DEBUG_PATH, os.path.basename(image_path).split('.')[0] + '_mask.png')
            # cv2.imwrite(str(display_path), (edge_visib.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        try:
            vertex = item['vertex']
            num_vertex = vertex.shape[0]
            if num_vertex < self.conf.sample_vertex_num:
                expand_num = self.conf.sample_vertex_num // num_vertex + 1
                vertex = vertex.unsqueeze(0).expand(expand_num, -1, -1).reshape(-1, 3)
                vertex = vertex[:self.conf.sample_vertex_num]
            else:
                # vertex, _ = sample_farthest_points(vertex[None], K=self.conf.sample_vertex_num)
                # vertex = vertex[0]
                # import ipdb
                # ipdb.set_trace()
                step = num_vertex // self.conf.sample_vertex_num
                vertex = vertex[::step, :]
                vertex = vertex[:self.conf.sample_vertex_num, :]
        except ValueError:
            import ipdb;
            ipdb.set_trace();

        data = {
            'image': image,
            # 'mask_visib': mask_visib,
            # 'edge_visib': edge_visib,
            # 'edge': edge,
            'camera': camera,
            'body2view_pose': body2view_pose,
            'aligned_vertex': vertex,
            'gt_body2view_pose': gt_body2view_pose,
            'closest_template_views': closest_template_views,
            'closest_orientations_in_body': closest_orientations_in_body,
            'diameter': diameter,
            'image_path': image_path,
            'obj_name': obj_id,
            'output_name': output_name,
            'OPT': item['OPT'],
            'sysmetric': False
        }

        if self.conf.img_aug == True and self.split == 'train':
            data['image_aug'] = aug_image

        return data

    def __len__(self):
        return len(self.items)


class _Dataset_test(_Dataset):
    def __init__(self, conf, split):
        super().__init__(conf, split)
        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        # logger.info(f'Sampling new images with seed {seed}')
        # num = self.conf[self.split + '_num_per_obj']
        self.items = []
        slices = []
        for slice in self.slices:
            if slice == 'all':
                seq_dir = os.path.join(self.root, slice[0])
                seq_list = os.listdir(seq_dir)
                for seq_name in seq_list:
                    slices.append(slice)
            else:
                slices.append((slice))

        for slice in tqdm(slices):
            data_dir = os.path.join(self.root, slice)
            K_path = os.path.join(data_dir, 'K.txt')
            K = np.loadtxt(K_path, dtype=np.float32)
            for obj_name in self.obj_names:
                img_dir = os.path.join(data_dir, obj_name)
                if not Path(img_dir).exists():
                    print("Warning: {} is not exist!".format(img_dir))
                    continue
                pose_path = os.path.join(img_dir, 'pose.txt')
                img_lists = glob.glob(img_dir + '/*.png', recursive=True)
                pose_txt = np.loadtxt(pose_path)
                image_paths = {}
                for image_path in img_lists:
                    img_id = int(image_path.split('/')[-1].split('.')[0])
                    image_paths[img_id] = image_path
                for img_id in range(len(img_lists)):
                    pose = pose_txt[img_id]
                    image_path = image_paths[img_id]
                    body2view_R = np.array(pose[:9], dtype=np.float32)
                    body2view_t = np.array(pose[9:], dtype=np.float32) * self.geometry_unit_in_meter
                    output_name = slice + "_" + obj_name + "_" + os.path.basename(image_path).split('.')[0]
                    skip = np.random.randint(1, self.conf.skip_frame + 1)
                    # if np.random.rand() < 0.5:
                    #     skip *= -1
                    last_img_id = img_id - skip
                    last_img_id = min(max(last_img_id, 0), len(pose_txt) - 1)
                    last_image_path = image_paths[last_img_id]
                    last_pose = pose_txt[last_img_id]
                    last_body2view_R = np.array(last_pose[:9], dtype=np.float32)
                    last_body2view_t = np.array(last_pose[9:], dtype=np.float32) * self.geometry_unit_in_meter
                    item = {'slice': slice, 'obj_id': obj_name, 'img_id': img_id, 'last_img_id': last_img_id,
                            'image_path': image_path, 'K': K, 'body2view_R': body2view_R, 'body2view_t': body2view_t,
                            'last_image_path': last_image_path, 'last_body2view_R': last_body2view_R,
                            'last_body2view_t': last_body2view_t, 'output_name': output_name, 'OPT': False,
                            'end': True if img_id == len(img_lists) - 1 else False}
                    self.items.append(item)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item['image_path']
        img_id = item['img_id']
        slice = item['slice']
        last_img_id = item['last_img_id']
        output_name = item['output_name']
        ori_image = read_image(image_path, self.conf.grayscale)
        obj_id = item['obj_id']
        body2view_R = item['body2view_R'].reshape(3, 3)
        body2view_t = item['body2view_t']
        gt_body2view_pose = Pose.from_Rt(body2view_R, body2view_t)
        K = item['K']
        intrinsic_param = torch.tensor([ori_image.shape[1], ori_image.shape[0],
                                        K[0], K[4], K[2], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)
        orientations_in_body = self.orientations_in_body[obj_id]
        template_views = self.template_views[obj_id]
        n_sample = self.num_sample_contour_points[obj_id]
        diameter = self.diameters[obj_id]

        last_body2view_R = item['last_body2view_R'].reshape(3, 3)
        last_body2view_t = item['last_body2view_t']
        body2view_pose = Pose.from_Rt(last_body2view_R, last_body2view_t)

        # get closest template view
        indices = get_closest_k_template_view_index(body2view_pose,
                                                    orientations_in_body,
                                                    self.conf.get_top_k_template_views)
        closest_template_views = torch.stack([template_views[ind * n_sample:(ind + 1) * n_sample, :]
                                              for ind in indices])
        closest_orientations_in_body = orientations_in_body[indices]

        # calc bbox
        data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])

        # read image
        image, camera = self.read_image(image_path, self.conf, ori_camera, bbox2d.numpy().copy(), ori_image,
                                        img_aug=self.conf.img_aug if self.split == 'train' else False)

        # read mask
        # mask_path = item['mask_path']
        # mask_visib_path = item['mask_visib_path']
        # edge_visib, edge, mask_visib = self.read_mask(mask_path, mask_visib_path, bbox2d.numpy().copy(), self.conf)

        # if self.conf.change_background:
        #     image = self.change_background(idx, image, mask_visib)

        # check
        if self.conf.debug_check_display:
            data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, camera)
            display_image = draw_correspondence_lines_in_image((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                                               data_lines['centers_in_image'],
                                                               data_lines['centers_valid'],
                                                               data_lines['normals_in_image'], 10)
            display_path = Path(os.path.basename(image_path))
            cv2.imwrite(str(display_path), display_image)

            # display_path = Path(DEBUG_PATH, os.path.basename(image_path).split('.')[0] + '_mask.png')
            # cv2.imwrite(str(display_path), (edge_visib.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        try:
            vertex = self.vertices[obj_id]
        except ValueError:
            import ipdb;
            ipdb.set_trace();

        data = {
            'image': image,
            'img_id': img_id,
            'last_img_id': last_img_id,
            'gt_body2view_pose': gt_body2view_pose,
            'orientations_in_body': orientations_in_body,
            'template_views': template_views,
            'n_sample': n_sample,
            # 'mask_visib': mask_visib,
            # 'edge_visib': edge_visib,
            # 'edge': edge,
            'camera': camera,
            # 'body2view_pose': body2view_pose,
            # 'closest_template_views': closest_template_views,
            # 'closest_orientations_in_body': closest_orientations_in_body,
            'aligned_vertex': vertex,
            'diameter': diameter,
            'image_path': image_path,
            'slice_name': slice,
            'obj_name': obj_id,
            'output_name': output_name,
            'OPT': item['OPT'],
            'end': item['end'],
            'sysmetric': False
        }

        return data