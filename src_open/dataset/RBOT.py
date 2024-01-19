import json
import os
from pathlib import Path
import glob
import cv2

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


class RBOT(BaseDataset):
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
        'random_sample': True,

        # 'normal_line_length': 32,
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
        'num_correspondence_lines': 200, 

        # 'image_width': 640,
        # 'image_height': 480,
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

        # 'min_visib_fract': 0.9,
        # 'min_px_count_visib': 3600,

        'debug_check_display': False
    }

    strict_conf = False

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        # assert split != 'test', 'Not supported'
        if split == 'train' or split == 'val':
            return _Dataset(self.conf, split)
        elif split == 'test':
            return _Dataset_test(self.conf, split)
        else:
            raise NotImplementedError

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(conf.dataset_dir)
        self.obj_names = conf.get(split + '_obj_names')
        self.slices = conf.get(split + '_slices')
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

        obj_template_view_paths = []
        for obj_name in self.obj_names:
            preprocess_path = os.path.join(self.root, obj_name, 'pre_render', obj_name + '.pkl')
            obj_template_view_paths.append(preprocess_path)
        self.num_sample_contour_points, self.template_views, self.orientations_in_body = \
            read_template_data(self.obj_names, obj_template_view_paths)

        for obj_name in self.obj_names:
            num_sample_contour_points = self.num_sample_contour_points[obj_name]
            assert(num_sample_contour_points % conf.num_correspondence_lines == 0)
            sample_step = num_sample_contour_points // conf.num_correspondence_lines
            template_views = self.template_views[obj_name]
            template_views = template_views.reshape(-1, num_sample_contour_points, template_views.shape[1])
            template_views = template_views[:, ::sample_step, :]
            self.num_sample_contour_points[obj_name] = conf.num_correspondence_lines
            self.template_views[obj_name] = template_views.reshape(-1, 8)

        self.vertices = {}
        self.diameters = {}
        for obj_name in self.obj_names:
            obj_path = os.path.join(self.root, obj_name, obj_name + '.obj')
            assert '.ply' in obj_path or '.obj' in obj_path
            if '.obj' in obj_path:
                vert, faces_idx, _ = load_obj(obj_path)
                face = faces_idx.verts_idx
            if '.ply' in obj_path:
                vert, face = load_ply(obj_path)
            self.vertices[obj_name] = vert * conf.geometry_unit_in_meter
            mesh = Meshes(verts=[vert * conf.geometry_unit_in_meter], faces=[face])
            bbox = mesh.get_bounding_boxes()
            diameter = bbox[..., 1] - bbox[..., 0]
            diameter, _ = torch.max(diameter, dim=1)
            self.diameters[obj_name] = diameter[0]

        # scene_gt_anno_path = os.path.join(self.root, 'YCB_Video_Dataset/image_sets/cls2seq_mapping.json')
        # with open(cls2seq_mapping_path, 'r', encoding='utf8') as fp:
        #     cls2seq_mapping = json.load(fp)
        # if split == 'train':
        #     self.cls2seq_mapping = cls2seq_mapping[split]
        # if split == 'val':
        #     self.cls2seq_mapping = cls2seq_mapping['test']
        # self.cls2seq_mapping['002_master_chef_can']
        if split != 'test':
            self.sample_new_items(conf.seed)

    # not used!
    def sample_new_items(self, seed):
        logger.info(f'Sampling new images with seed {seed}')
        set_seed(seed)
        num = self.conf[self.split + '_num_per_obj']
        self.items = []
        obj_items = {}
        slices = []
        for slice in self.slices:
            if slice == 'all':
                seq_dir = os.path.join(self.root, slice[0])
                seq_list = os.listdir(seq_dir)
                for seq_name in seq_list:
                    slices.append(slice)
            else:
                slices.append((slice))

        K_path = os.path.join(self.root, 'camera_calibration.txt')
        K = np.loadtxt(K_path, dtype=np.float32)
        K = np.array([K[0], 0, K[2], 0, K[1], K[3], 0, 0, 1], dtype=np.float32)
        pose_path = os.path.join(self.root, 'poses_first.txt')
        pose_txt = np.loadtxt(pose_path)

        for slice in tqdm(slices):
            for obj_name in self.obj_names:
                data_dir = os.path.join(self.root, obj_name)
                img_dir = os.path.join(data_dir, 'frames')
                if not Path(img_dir).exists():
                    print("Warning: {} is not exist!".format(img_dir))
                    continue
                img_lists = glob.glob(img_dir + '/' + slice + '*.png', recursive=True)
                image_paths = {}
                for image_path in img_lists:
                    img_id = int(image_path.split('/')[-1].split('.')[0][-4:])
                    image_paths[img_id] = image_path
                for img_id, image_path in image_paths.items():
                    pose = pose_txt[img_id]
                    body2view_R = np.array(pose[:9], dtype=np.float32)
                    body2view_t = np.array(pose[9:], dtype=np.float32) * self.geometry_unit_in_meter
                    output_name = obj_name + "_" + os.path.basename(image_path).split('.')[0]
                    skip = np.random.randint(1, self.conf.skip_frame+1)
                    # if np.random.rand() < 0.5:
                    #     skip *= -1
                    last_img_id = img_id - skip
                    last_img_id = min(max(last_img_id, 0), len(pose_txt)-1)
                    last_image_path = image_paths[last_img_id]
                    last_pose = pose_txt[last_img_id]
                    last_body2view_R = np.array(last_pose[:9], dtype=np.float32)
                    last_body2view_t = np.array(last_pose[9:], dtype=np.float32) * self.geometry_unit_in_meter
                    item = {'obj_id': obj_name, 'img_id': img_id, 'image_path': image_path, 'K': K,
                            'body2view_R': body2view_R, 'body2view_t': body2view_t, 'last_image_path': last_image_path,
                            'last_body2view_R': last_body2view_R, 'last_body2view_t': last_body2view_t,
                            'output_name': output_name, 'OPT': False}

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

    # not used!
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
        orientations_in_body = self.orientations_in_body[obj_id]
        template_views = self.template_views[obj_id]
        n_sample = self.num_sample_contour_points[obj_id]
        diameter = self.diameters[obj_id]

        # generate offset to ground truth pose
        if (img_id == 0) or (self.split == 'train' and self.conf.train_offset) or \
                (self.split == 'val' and self.conf.val_offset):  # self.split == 'train' or self.conf.val_offset:
            random_aa, random_t = generate_random_aa_and_t(self.min_offset_angle, self.max_offset_angle,
                                                           self.min_offset_translation, self.max_offset_translation)
            random_pose = Pose.from_aa(random_aa, random_t)
            body2view_pose = gt_body2view_pose @ random_pose[0]
        else:
            last_body2view_R = item['last_body2view_R'].reshape(3, 3)
            last_body2view_t = item['last_body2view_t']
            body2view_pose = Pose.from_Rt(last_body2view_R, last_body2view_t)
            # raise NotImplementedError

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
            ori_image_with_background = self.change_background(idx, ori_image, ori_mask, self.conf.change_background_thres)
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
            'sysmetric': False
        }

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
            
        K_path = os.path.join(self.root, 'camera_calibration.txt')
        K = np.loadtxt(K_path, dtype=np.float32)
        K = np.array([K[0], 0, K[2], 0, K[1], K[3], 0, 0, 1], dtype=np.float32)
        pose_path = os.path.join(self.root, 'poses_first.txt')
        pose_txt = np.loadtxt(pose_path)

        for slice in tqdm(slices):
            for obj_name in self.obj_names:
                data_dir = os.path.join(self.root, obj_name)
                img_dir = os.path.join(data_dir, 'frames')
                if not Path(img_dir).exists():
                    print("Warning: {} is not exist!".format(img_dir))
                    continue
                    
                img_full_lists = glob.glob(img_dir + '/' + slice + '*.png', recursive=True)
                image_full_paths = {}
                for image_path in img_full_lists:
                    img_id = int(image_path.split('/')[-1].split('.')[0][-4:])
                    image_full_paths[img_id] = image_path
                
                image_order_paths = []
                for img_id in range(len(img_full_lists)):
                    image_order_paths.append(image_full_paths[img_id])

                skip = self.conf.skip_frame
                num_img = len(image_order_paths)
                image_paths = image_order_paths[0:num_img:skip]
                pose_txt = pose_txt[0:num_img:skip]

                for img_id in range(len(image_paths)):  
                    pose = pose_txt[img_id]
                    image_path = image_paths[img_id]
                    body2view_R = np.array(pose[:9], dtype=np.float32)
                    body2view_t = np.array(pose[9:], dtype=np.float32) * self.geometry_unit_in_meter
                    output_name = slice + '_' + obj_name + '_' + os.path.basename(image_path).split('.')[0]
                    # if np.random.rand() < 0.5:
                    #     skip *= -1
                    last_img_id = img_id - 1 
                    last_img_id = min(max(last_img_id, 0), len(pose_txt)-1)
                    last_image_path = image_paths[last_img_id]
                    last_pose = pose_txt[last_img_id]
                    last_body2view_R = np.array(last_pose[:9], dtype=np.float32)
                    last_body2view_t = np.array(last_pose[9:], dtype=np.float32) * self.geometry_unit_in_meter
                    item = {'slice': slice, 'obj_id': obj_name, 'img_id': img_id, 'last_img_id': last_img_id,
                            'image_path': image_path, 'K': K, 'body2view_R': body2view_R, 'body2view_t': body2view_t,
                            'last_image_path': last_image_path, 'last_body2view_R': last_body2view_R,
                            'last_body2view_t': last_body2view_t, 'output_name': output_name, 'OPT': False,
                            'end': True if img_id == len(image_paths)-1 else False}
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