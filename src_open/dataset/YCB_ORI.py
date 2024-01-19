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
from pytorch3d.transforms import quaternion_to_matrix

logger = logging.getLogger(__name__)


class YCB_ORI(BaseDataset):
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
        'geometry_unit_in_meter': 1,  # must equal to the geometry_unit_in_meter of preprocess
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
            raise NotImplementedError
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

        obj_names_file = os.path.join(self.root, 'image_sets', 'classes.txt')
        obj_names = np.loadtxt(obj_names_file, dtype=np.str)
        self.obj_name2class_index = {}
        for i, obj_name in enumerate(obj_names):
            self.obj_name2class_index[obj_name] = i + 1

        obj_template_view_paths = []
        obj_ids = []
        self.obj_ids = []
        for obj_name in self.obj_names:
            obj_id = str(self.obj_name2class_index[obj_name])
            preprocess_path = os.path.join(self.root, 'models', obj_name, 'pre_render', 'textured_simple.pkl')
            obj_template_view_paths.append(preprocess_path)
            obj_ids.append(obj_id)
            self.obj_ids.append(int(obj_id))
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
            obj_path = os.path.join(self.root, 'models', obj_name, 'textured_simple.ply')
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


class _Dataset_test(_Dataset):
    def __init__(self, conf, split):
        super().__init__(conf, split)
        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        import scipy
        logger.info(f'Sampling new images with seed {seed}')
        set_seed(seed)
        num = self.conf[self.split + '_num_per_obj']
        self.items = []
        obj_items = {}
        slices = []
        data_dir = os.path.join(self.root, 'data')

        if 'all' in self.slices:
            keyframe_txt = os.path.join(self.root, 'image_sets', 'keyframe.txt')
            frames = np.loadtxt(keyframe_txt, dtype=np.str)
        else:
            raise NotImplementedError
        
        i = 0
        add_num = 0
        # tmp_i = 0
        for frame in tqdm(frames):
            seq_name = frame.split('/')[0]
            seq_dir = os.path.join(data_dir, seq_name)
            image_name = frame.split('/')[1]
            image_path = os.path.join(seq_dir, image_name+'-color.png')
            meta_path = os.path.join(seq_dir, image_name+'-meta.mat')
            meta = scipy.io.loadmat(meta_path)
            
            K = meta['intrinsic_matrix']
            cls_indexes = meta['cls_indexes'][..., 0]
            gt_poses = {}
            for j, cls_idx in enumerate(cls_indexes):
                gt_poses[cls_idx] = meta['poses'][..., j]
            posecnn_meta_path = os.path.join(self.root, 'result_posecnn', str(i).zfill(6)+'.mat')
            posecnn_result = scipy.io.loadmat(posecnn_meta_path)
            poses = posecnn_result['poses']
            rois = posecnn_result['rois']
            obj_ids = rois[..., 1]

            # if seq_name == '0050':
            #     import ipdb
            #     ipdb.set_trace()

            for pose, obj_id in zip(poses, obj_ids):
                obj_id = int(obj_id)
                if obj_id in self.obj_ids and obj_id in cls_indexes:
                    try:
                        gt_pose = torch.from_numpy(gt_poses[obj_id]).float()
                    except KeyError:
                        import ipdb;
                        ipdb.set_trace();

                    obj_name = self.obj_names[self.obj_ids.index(obj_id)]
                    gt_body2view_R = gt_pose[:3, :3]
                    gt_body2view_t = gt_pose[:3, 3]
                    pose = torch.from_numpy(pose)
                    body2view_R = quaternion_to_matrix(pose[:4]).float()
                    body2view_t = pose[4:].float() * self.geometry_unit_in_meter
                    item = {'obj_id': obj_name, 'image_path': image_path,
                            'K': K, 'body2view_R': body2view_R, 'body2view_t': body2view_t,
                            'gt_body2view_R': gt_body2view_R, 'gt_body2view_t': gt_body2view_t}
                    item['output_name'] = f'{obj_name}_{seq_name}_{image_name}'
                    if obj_id not in obj_items.keys():
                        obj_items[obj_id] = []
                    obj_items[obj_id].append(item)
                    add_num += 1
            i += 1
            # if add_num>20:
            #     break

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

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item['image_path']
        output_name = item['output_name']
        ori_image = read_image(image_path, self.conf.grayscale)
        obj_id = item['obj_id']
        body2view_R = item['body2view_R'].reshape(3, 3)
        body2view_t = item['body2view_t']
        body2view_pose = Pose.from_Rt(body2view_R, body2view_t)
        gt_body2view_R = item['gt_body2view_R'].reshape(3, 3)
        gt_body2view_t = item['gt_body2view_t']
        gt_body2view_pose = Pose.from_Rt(gt_body2view_R, gt_body2view_t)
        K = item['K']
        # intrinsic_param = torch.tensor([ori_image.shape[1], ori_image.shape[0],
        #                                 K[0], K[4], K[2], K[5]], dtype=torch.float32)
        intrinsic_param = torch.tensor([ori_image.shape[1], ori_image.shape[0],
                                        K[0][0], K[1][1], K[0][2], K[1][2]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)
        orientations_in_body = self.orientations_in_body[obj_id]
        template_views = self.template_views[obj_id]
        n_sample = self.num_sample_contour_points[obj_id]
        diameter = self.diameters[obj_id]

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


        if False:
            data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, camera)
            display_image = draw_correspondence_lines_in_image((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                                               data_lines['centers_in_image'],
                                                               data_lines['centers_valid'],
                                                               data_lines['normals_in_image'], 10)
            display_path = 'test.png'# Path(os.path.basename(image_path))
            cv2.imwrite(str(display_path), display_image)

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
            'OPT': False,
            'sysmetric': False
        }

        if self.conf.img_aug == True and self.split == 'train':
            data['image_aug'] = aug_image

        return data
    
    def __len__(self):
        return len(self.items)