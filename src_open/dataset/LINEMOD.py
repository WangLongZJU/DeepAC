import json
import os
from pathlib import Path
import glob
import cv2

from .base_dataset import BaseDataset
import torch
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
from pytorch3d.transforms import axis_angle_to_matrix

logger = logging.getLogger(__name__)

class LINEMOD(BaseDataset):
    default_conf = {
        'dataset_dir': '',
        'background_image_dir': '',

        'train_obj_names': [''],
        'train_slices': [''],
        'val_obj_names': [''],
        'val_slices': [''],
        'test_obj_names': [''],
        'test_slices': [''],
        'train_num_per_obj': 2000,
        'val_num_per_obj': 2000,
        'test_num_per_obj': 2000,
        'random_sample': True,

        # 'normal_line_length': 32,
        'get_top_k_template_views': 1,
        'geometry_unit_in_meter': 1.0,  # must equal to the geometry_unit_in_meter of preprocess
        'offset_angle_step': 5.0,
        'min_offset_angle': 5.0,  # 5.0,
        'max_offset_angle': 15.0,  # 25.0,
        'offset_translation_step': 0.01,
        'min_offset_translation': 0.005,  # 0.01,# 0.01,  # meter
        'max_offset_translation': 0.015,  # 0.025,# 0.03,  # meter
        'val_offset': True,
        'train_offset': True,

        # 'image_width': 640,
        # 'image_height': 480,
        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': False,
        'crop_border': None,
        'pad': None,
        'change_background': False,
        'img_aug': False,
        'seed': 0,
        'sample_vertex_num': 500,

        # 'min_visib_fract': 0.9,
        # 'min_px_count_visib': 3600,

        'debug_check_display': False
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        # assert split != 'test', 'Not supported'
        # assert split == 'test'
        # if split == 'train' or split == 'val':
        #     return _Dataset(self.conf, split)
        # elif split == 'test':
        #     return _Dataset_test(self.conf, split)
        # else:
        #     raise NotImplementedError

        return _Dataset(self.conf, split)

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(conf.dataset_dir)
        self.obj_names = conf.get(split + '_obj_names')
        self.slices = conf.get(split + '_slices')
        self.conf, self.split = conf, split

        self.K = np.array([572.4114, 0., 325.2611,
                           0., 573.57043, 242.04899,
                           0., 0., 1.], dtype=np.float32)

        self.K = torch.from_numpy(self.K)

        self.geometry_unit_in_meter = float(conf.geometry_unit_in_meter)
        self.min_offset_angle = float(conf.min_offset_angle)
        self.max_offset_angle = float(conf.max_offset_angle)
        self.min_offset_translation = float(conf.min_offset_translation)
        self.max_offset_translation = float(conf.max_offset_translation)
        # self.image_width = int(conf.image_width)
        # self.image_height = int(conf.image_height)

        # check background if change_background is True
        if conf.change_background is True:
            self.background_image_dir = Path(conf.background_image_dir, 'JPEGImages')
            assert self.background_image_dir.exists()
            self.background_image_path = np.stack(os.listdir(str(self.background_image_dir)))

        obj_template_view_paths = []
        self.obj_ids = []
        for obj_name in self.obj_names:
            preprocess_path = os.path.join(self.root, 'linemod', obj_name, 'pre_render', obj_name+'.pkl')
            obj_template_view_paths.append(preprocess_path)
            self.obj_ids.append(obj_name)
        self.num_sample_contour_points, self.template_views, self.orientations_in_body = \
            read_template_data(self.obj_names, obj_template_view_paths)

        self.vertices = {}
        for obj_name in self.obj_names:
            obj_path = os.path.join(self.root, 'linemod', obj_name, obj_name+'.obj')
            assert '.ply' in obj_path or '.obj' in obj_path
            if '.obj' in obj_path:
                vert, faces_idx, _ = load_obj(obj_path)
                face = faces_idx.verts_idx
            if '.ply' in obj_path:
                vert, face = load_ply(obj_path)
            self.vertices[obj_name] = vert * conf.geometry_unit_in_meter

        self.diameter = {
            'cat': 0.152633,
            'ape': 0.0974298,
            'benchvise': 0.286908,
            'bowl': 0.171185,
            'cam': 0.171593,
            'can': 0.193416,
            'cup': 0.125961,
            'driller': 0.259425,
            'duck': 0.107131,
            'eggbox': 0.176364,
            'glue': 0.164857,
            'holepuncher': 0.148204,
            'iron': 0.303153,
            'lamp': 0.285155,
            'phone': 0.208394
        }

        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images with seed {seed}')
        num = self.conf[self.split + '_num_per_obj']
        self.items = []
        obj_items = {}

        for obj_name in tqdm(self.obj_names):
            data_dir = os.path.join(self.root, 'linemod', obj_name)
            txt_path = os.path.join(data_dir, self.split+'.txt')
            image_list = np.loadtxt(txt_path, dtype=np.str)
            for i, image_name in enumerate(image_list):
                image_name = image_name.split('/')[-1]
                image_idx = int(image_name.split('.')[0])
                image_path = os.path.join(data_dir, 'JPEGImages', image_name)
                # mask_path = os.path.join(data_dir, 'mask', image_name[2:].replace('jpg', 'png'))
                pose_path = os.path.join(data_dir, 'pose', 'pose'+str(image_idx)+'.npy')
                pose = np.load(pose_path)
                body2view_R = torch.from_numpy(pose[:3, :3]).float()
                body2view_t = torch.from_numpy(pose[:, 3]) * self.geometry_unit_in_meter
                output_name = obj_name + "_" + os.path.basename(image_path).split('.')[0]
                item = {'obj_id': str(obj_name), 'image_path': image_path, 'K': self.K,
                        'body2view_R': body2view_R, 'body2view_t': body2view_t, 'output_name': output_name,
                        'sysmetric': True if obj_name in ['eggbox', 'glue'] else False}
                if self.split == 'test':
                    pvnet_result_path = os.path.join(self.root, 'LinemodTest', obj_name, str(i+1)+'.npy')
                    pvnet_result = np.load(pvnet_result_path, allow_pickle=True).item()
                    init_pose = pvnet_result['x_ini']
                    init_body2view_axis_angle = torch.from_numpy(init_pose[:3]).float()
                    item['init_body2view_R'] = axis_angle_to_matrix(init_body2view_axis_angle)
                    item['init_body2view_t'] = torch.from_numpy(init_pose[3:]).float() * self.geometry_unit_in_meter
                if obj_name not in obj_items.keys():
                    obj_items[obj_name] = []
                obj_items[obj_name].append(item)

        for key in obj_items:
            items = obj_items[key]
            items = np.stack(items)
            if len(items) > num:
                if self.conf.random_sample:
                    selected = np.random.RandomState(seed).choice(
                        len(items), num, replace=False)
                    items = items[selected]
                else:
                    items = items[:num]
            self.items.extend(items)

        if self.conf.change_background is True:
            selected = np.random.RandomState(seed).choice(
                len(self.background_image_path), len(self.items), replace=False)
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
                    # img, scales = resize(img, conf.resize, fn=max)
                    try:
                        img, scales = resize(img, conf.resize, fn=max)
                    except cv2.error:
                        import ipdb;
                        ipdb.set_trace();
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

        if conf.img_aug:
            img_aug = self.image_aug(img)
        else:
            img_aug = img
        img_aug = img_aug.astype(np.float32)

        return numpy_image_to_torch(img_aug), camera

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

    def change_background(self, idx, image, mask):

        if np.random.rand() < 0.5:
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
        ori_image = read_image(image_path, self.conf.grayscale)
        obj_id = item['obj_id']
        output_name = item['output_name']
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
        diameter = self.diameter[obj_id] # * self.geometry_unit_in_meter

        # generate offset to ground truth pose
        if (self.split == 'train' and self.conf.train_offset) or \
           (self.split == 'val' and self.conf.val_offset):
            random_aa, random_t = generate_random_aa_and_t(self.min_offset_angle, self.max_offset_angle,
                                                           self.min_offset_translation, self.max_offset_translation)
            random_pose = Pose.from_aa(random_aa, random_t)
            body2view_pose = gt_body2view_pose @ random_pose[0]
        else:
            body2view_pose = Pose.from_Rt(item['init_body2view_R'], item['init_body2view_t'])

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

        if self.conf.change_background and self.split == 'train':
            ori_mask = self.draw_mask(template_views, gt_body2view_pose, orientations_in_body,
                                      n_sample, ori_camera, ori_image)
            ori_image_with_background = self.change_background(idx, ori_image, ori_mask)
            image = self.transform_img(ori_image_with_background, bbox2d.numpy().copy(), self.conf)

        # read mask
        # mask_path = item['mask_path']
        # # mask_visib_path = item['mask_visib_path']
        # edge_visib, edge, mask_visib = self.read_mask(mask_path, mask_path, bbox2d.numpy().copy(), self.conf)

        # check
        if self.conf.debug_check_display:
            data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, camera)
            display_image = draw_correspondence_lines_in_image((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                                               data_lines['centers_in_image'],
                                                               data_lines['centers_valid'],
                                                               data_lines['normals_in_image'], 1)
            display_path = Path(os.path.basename(image_path))
            cv2.imwrite(str(display_path), display_image)

            # display_path = Path(DEBUG_PATH, os.path.basename(image_path).split('.')[0] + '_mask.png')
            # cv2.imwrite(str(display_path), (edge_visib.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        try:
            vertex = self.vertices[obj_id]
            num_vertex = vertex.shape[0]
            if num_vertex < self.conf.sample_vertex_num:
                expand_num = self.conf.sample_vertex_num // num_vertex + 1
                vertex = vertex.unsqueeze(0).expand(expand_num, -1, -1).reshape(-1, 3)
                vertex = vertex[:self.conf.sample_vertex_num]
            else:
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
            'sysmetric': item['sysmetric']
        }

        return data

    def __len__(self):
        return len(self.items)