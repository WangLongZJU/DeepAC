import json
import os
from pathlib import Path
import glob
import cv2
import time

from .base_dataset import BaseDataset, set_seed
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

logger = logging.getLogger(__name__)


class _Dataset_ICBIN(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(conf.dataset_dir, 'icbin')

        self.obj_names = conf.get('icbin_' + split + '_obj_names')  # split: train/val
        self.pbr_slices = conf.get('icbin_' + split + '_pbr_slices')  # split: train/val
        self.real_slices = conf.get('icbin_' + split + '_real_slices')  # split: train/val

        self.conf, self.split = conf, split

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

        # # read model_json
        # models_json_path = os.path.join(self.root, 'models', 'models_info.json')
        # with open(models_json_path, 'r', encoding='utf8') as fp:
        #     self.models_json = json.load(fp)

        obj_template_view_paths = []
        obj_ids = []
        self.obj_ids = []
        assert self.obj_names != 'none'

        for obj_name in self.obj_names:
            obj_id = str(int(obj_name.split('_')[-1]))
            preprocess_path = os.path.join(self.root, 'models', obj_name, 'pre_render', obj_name+'.pkl')
            obj_template_view_paths.append(preprocess_path)
            # add sub_dataset prefix 
            obj_ids.append('icbin_' + obj_id)
            self.obj_ids.append(int(obj_id))
        self.num_sample_contour_points, self.template_views, self.orientations_in_body = \
            read_template_data(obj_ids, obj_template_view_paths)

        self.vertices = {}
        self.diameters = {}
        for obj_name in self.obj_names:
            obj_path = os.path.join(self.root, 'models', obj_name+'.ply')
            assert '.ply' in obj_path or '.obj' in obj_path
            if '.obj' in obj_path:
                vert, faces_idx, _ = load_obj(obj_path)
                face = faces_idx.verts_idx
            if '.ply' in obj_path:
                vert, face = load_ply(obj_path)            
            obj_id = str(int(obj_name.split('_')[-1]))
            self.vertices['icbin_' + obj_id] = vert * conf.geometry_unit_in_meter
            mesh = Meshes(verts=[vert * conf.geometry_unit_in_meter], faces=[face])
            bbox = mesh.get_bounding_boxes()
            diameter = bbox[..., 1] - bbox[..., 0]
            diameter, _ = torch.max(diameter, dim=1)
            self.diameters['icbin_' + obj_id] = diameter[0]

        # if split != 'test':
        #     self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images with seed {seed}')
        set_seed(seed)
        pbr_slices = []
        real_slices = []

        # add slices, including pbr_slices and real_slices
        # pbr_slices
        seq_dir = os.path.join(self.root, 'train_pbr')
        if self.pbr_slices == 'none':
            pbr_slices = []
        else:
            seq_list = os.listdir(seq_dir)
            seq_list.sort()
            if self.pbr_slices == 'all':
                pbr_slices = seq_list
            elif self.pbr_slices == 'train_split':
                pbr_slices = seq_list[:int(0.7*len(seq_list))]
            elif self.pbr_slices == 'val_split':
                pbr_slices = seq_list[int(0.7*len(seq_list)):]
            else:
                raise NotImplementedError

        # real_slices
        seq_dir = os.path.join(self.root, 'test')
        if self.real_slices == 'none':
            real_slices = []
        else:
            seq_list = os.listdir(seq_dir)
            seq_list.sort()
            if self.real_slices == 'all':
                real_slices = seq_list
            else:
                raise NotImplementedError

        # ---------------------------------------------------
        self.items = []
        obj_items = {}
        # pbr_slices
        for pbr_slice in tqdm(pbr_slices):
            data_dir = os.path.join(self.root, 'train_pbr', pbr_slice)
            scene_anno_path = os.path.join(data_dir, 'scene_gt.json')
            with open(scene_anno_path, 'r', encoding='utf8') as fp:
                scene_anno = json.load(fp)
            scene_meta_anno_path = os.path.join(data_dir, 'scene_gt_info.json')
            with open(scene_meta_anno_path, 'r', encoding='utf8') as fp:
                scene_meta_anno = json.load(fp)
            scene_camera_anno_path = os.path.join(data_dir, 'scene_camera.json')    
            with open(scene_camera_anno_path, 'r', encoding='utf8') as fp:
                scene_camera_anno = json.load(fp)

            for image_id in scene_anno:
                image_anno = scene_anno[image_id]
                meta_anno = scene_meta_anno[image_id]
                K = np.array(scene_camera_anno[image_id]['cam_K'], dtype=np.float32)
                
                for i, (obj_anno, meta) in enumerate(zip(image_anno, meta_anno)):
                    obj_id = obj_anno['obj_id']
                    px_count_visib = meta['px_count_visib']
                    visib_fract = meta['visib_fract']
                    if (obj_id in self.obj_ids) and (visib_fract > self.conf.min_visib_fract) and \
                            (px_count_visib > self.conf.min_px_count_visib):
                        image_name = image_id.zfill(6) + '.jpg' 
                        mask_name = image_id.zfill(6) + '_' + str(i).zfill(6) + '.png'
                        image_path = os.path.join(data_dir, 'rgb', image_name)
                        mask_path = os.path.join(data_dir, 'mask', mask_name)
                        mask_visib_path = os.path.join(data_dir, 'mask_visib', mask_name)
                        body2view_R = np.array(obj_anno['cam_R_m2c'], dtype=np.float32)
                        body2view_t = np.array(obj_anno['cam_t_m2c'], dtype=np.float32) * self.geometry_unit_in_meter
                        output_name = 'pbr_' + pbr_slice + '_icbin_' + str(obj_id) + '_' + os.path.basename(image_path).split('.')[0]
                        
                        orientations_in_body = self.orientations_in_body['icbin_' + str(obj_id)]
                        template_views = self.template_views['icbin_' + str(obj_id)]
                        n_sample = self.num_sample_contour_points['icbin_' + str(obj_id)]
                        diameter = self.diameters['icbin_' + str(obj_id)]
                        vertex = self.vertices['icbin_' + str(obj_id)]
                        
                        item = {'slice': pbr_slice, 'obj_id': 'icbin_' + str(obj_id), 'img_id': image_id,
                                'image_path': image_path, 'K': K, 'body2view_R': body2view_R, 'body2view_t': body2view_t,
                                'mask_path': mask_path, 'mask_visib_path': mask_visib_path, 'output_name': output_name,
                                'orientations_in_body': orientations_in_body, 'template_views': template_views, 
                                'n_sample': n_sample, 'diameter': diameter, 'vertex': vertex, 'OPT': False}
                        
                        if 'icbin_' + str(obj_id) not in obj_items.keys():
                            obj_items['icbin_' + str(obj_id)] = []
                        obj_items['icbin_' + str(obj_id)].append(item)

        # real_slices
        for real_slice in tqdm(real_slices):
            data_dir = os.path.join(self.root, 'test', real_slice)
            scene_anno_path = os.path.join(data_dir, 'scene_gt.json')
            with open(scene_anno_path, 'r', encoding='utf8') as fp:
                scene_anno = json.load(fp)
            scene_meta_anno_path = os.path.join(data_dir, 'scene_gt_info.json')
            with open(scene_meta_anno_path, 'r', encoding='utf8') as fp:
                scene_meta_anno = json.load(fp)
            scene_camera_anno_path = os.path.join(data_dir, 'scene_camera.json')
            with open(scene_camera_anno_path, 'r', encoding='utf8') as fp:
                scene_camera_anno = json.load(fp)
            
            for image_id in scene_anno:
                image_anno = scene_anno[image_id]
                meta_anno = scene_meta_anno[image_id]
                K = np.array(scene_camera_anno[image_id]['cam_K'], dtype=np.float32)
                for i, (obj_anno, meta) in enumerate(zip(image_anno, meta_anno)):
                    obj_id = obj_anno['obj_id']
                    px_count_visib = meta['px_count_visib']
                    visib_fract = meta['visib_fract']
                    if (obj_id in self.obj_ids) and (visib_fract > self.conf.min_visib_fract) and \
                            (px_count_visib > self.conf.min_px_count_visib):
                        image_name = image_id.zfill(6) + '.png' 
                        mask_name = image_id.zfill(6) + '_' + str(i).zfill(6) + '.png'
                        image_path = os.path.join(data_dir, 'rgb', image_name)
                        mask_path = os.path.join(data_dir, 'mask', mask_name)
                        mask_visib_path = os.path.join(data_dir, 'mask_visib', mask_name)
                        body2view_R = np.array(obj_anno['cam_R_m2c'], dtype=np.float32)
                        body2view_t = np.array(obj_anno['cam_t_m2c'], dtype=np.float32) * self.geometry_unit_in_meter
                        output_name = 'real_' + real_slice + '_icbin_' + str(obj_id) + '_' + os.path.basename(image_path).split('.')[0]
                        
                        orientations_in_body = self.orientations_in_body['icbin_' + str(obj_id)]
                        template_views = self.template_views['icbin_' + str(obj_id)]
                        n_sample = self.num_sample_contour_points['icbin_' + str(obj_id)]
                        diameter = self.diameters['icbin_' + str(obj_id)]
                        vertex = self.vertices['icbin_' + str(obj_id)]
                        
                        item = {'slice': real_slice, 'obj_id': 'icbin_' + str(obj_id), 'img_id': image_id,
                                'image_path': image_path, 'K': K, 'body2view_R': body2view_R, 'body2view_t': body2view_t,
                                'mask_path': mask_path, 'mask_visib_path': mask_visib_path, 'output_name': output_name,
                                'orientations_in_body': orientations_in_body, 'template_views': template_views, 
                                'n_sample': n_sample, 'diameter': diameter, 'vertex': vertex, 'OPT': False}
                        
                        if 'icbin_' + str(obj_id) not in obj_items.keys():
                            obj_items['icbin_' + str(obj_id)] = []
                        obj_items['icbin_' + str(obj_id)].append(item)

        num = self.conf[self.split + '_num_per_obj']
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

        # image_info_file = os.path.join('./image.txt')
        # with open(image_info_file, 'w') as f:
        #     content = '\n'.join(image_paths)
        #     f.write(content)
        # mask_info_file = os.path.join('./mask.txt')
        # with open(mask_info_file, 'w') as f:
        #     content = '\n'.join(mask_paths)
        #     f.write(content)
        # import ipdb;
        # ipdb.set_trace();

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
        image, camera = self.read_image(image_path, self.conf, ori_camera, bbox2d.numpy().copy(), ori_image,
                                        img_aug=self.conf.img_aug if self.split == 'train' else False)

        if self.conf.change_background and self.split == 'train':
            ori_mask = self.draw_mask(template_views, gt_body2view_pose, orientations_in_body,
                                      n_sample, ori_camera, ori_image)
            ori_image_with_background = self.change_background(idx, ori_image, ori_mask)
            image = self.transform_img(ori_image_with_background, bbox2d.numpy().copy(), self.conf)

        # new_image = (new_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./test.png', new_image)

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

        return data

    def __len__(self):
        return len(self.items)