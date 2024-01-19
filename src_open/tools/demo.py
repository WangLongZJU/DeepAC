import os
import torch
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import cv2
import copy
import warnings
import pickle
import glob
from tqdm import tqdm

from ..utils.geometry.wrappers import Pose, Camera
from ..models import get_model
from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from ..dataset.utils import read_image, resize, numpy_image_to_torch, crop, zero_pad, get_imgaug_seq
from ..utils.utils import project_correspondences_line, get_closest_template_view_index,\
    get_closest_k_template_view_index, get_bbox_from_p2d
from ..models.deep_ac import calculate_basic_line_data

@torch.no_grad()
def main(cfg):

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'demo_cfg.yml')
    assert ('load_cfg' in cfg)
    # assert ('load_model' in cfg)
    assert (Path(cfg.load_cfg).exists())
    # assert (Path(cfg.load_model).exists())
    train_cfg = OmegaConf.load(cfg.load_cfg)
    data_conf = train_cfg.data
    logger.dump_cfg(train_cfg, 'train_cfg.yml')

    model = get_model(train_cfg.models.name)(train_cfg.models)
    ckpt = torch.load(cfg.load_model, map_location='cpu')
    if "pytorch-lightning_version" not in ckpt:
        warnings.warn(
            "Warning! Old .pth checkpoint is deprecated. "
            "Convert the checkpoint with tools/convert_old_checkpoint.py "
        )
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    logger.info("Loaded model weight from {}".format(cfg.load_model))
    model.cuda()
    model.eval()

    fore_learn_rate = cfg.fore_learn_rate
    back_learn_rate = cfg.back_learn_rate

    obj_name = cfg.obj_name
    data_dir = cfg.data_dir
    img_dir = os.path.join(data_dir, 'img')
    pose_path = os.path.join(img_dir, 'pose.txt')
    K_path = os.path.join(data_dir, 'K.txt')
    template_path = os.path.join(data_dir, obj_name, 'pre_render', f'{obj_name}.pkl')

    with open(template_path, "rb") as pkl_handle:
        pre_render_dict = pickle.load(pkl_handle)
    head = pre_render_dict['head']
    num_sample_contour_points = head['num_sample_contour_point']
    template_views = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
    orientations = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)

    K = torch.from_numpy(np.loadtxt(K_path)).type(torch.float32)
    poses = torch.from_numpy(np.loadtxt(pose_path)).type(torch.float32)
    init_pose = poses[0]
    init_R = init_pose[:9].reshape(3, 3)
    init_t = init_pose[9:] * cfg.geometry_unit_in_meter
    init_pose = Pose.from_Rt(init_R, init_t)
    img_lists = glob.glob(img_dir+'/*.png', recursive=True)
    img_lists.sort()
    
    def preprocess_image(img, bbox2d, camera):
        bbox2d[2:] += data_conf.crop_border * 2
        img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)

        scales = (1, 1)
        if isinstance(data_conf.resize, int):
            if data_conf.resize_by == 'max':
                # print('img shape', img.shape)
                # print('img path', image_path)
                img, scales = resize(img, data_conf.resize, fn=max)
            elif (data_conf.resize_by == 'min' or (data_conf.resize_by == 'min_if' and min(*img.shape[:2]) < data_conf.resize)):
                img, scales = resize(img, data_conf.resize, fn=min)
        elif len(data_conf.resize) == 2:
            img, scales = resize(img, list(data_conf.resize))
        if scales != (1, 1):
            camera = camera.scale(scales)

        img, = zero_pad(data_conf.pad, img)
        img = img.astype(np.float32)
        return numpy_image_to_torch(img), camera

    if cfg.output_video:
        video = cv2.VideoWriter(os.path.join(logger.log_dir, obj_name + ".avi"),  # 
                                cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, cfg.output_size)

    for i, img_path in enumerate(tqdm(img_lists)):
        ori_image = read_image(img_path)
        height, width = ori_image.shape[:2]
        intrinsic_param = torch.tensor([width, height, K[0], K[4], K[2], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)

        indices = get_closest_k_template_view_index(init_pose, orientations,
                                                    data_conf.get_top_k_template_views * data_conf.skip_template_view)
        closest_template_views = torch.stack([template_views[ind * num_sample_contour_points:(ind + 1) * num_sample_contour_points, :]
                                                for ind in indices[::data_conf.skip_template_view]])
        closest_orientations_in_body = orientations[indices[::data_conf.skip_template_view]]
        data_lines = project_correspondences_line(closest_template_views[0], init_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])
        img, camera = preprocess_image(ori_image, bbox2d.numpy().copy(), ori_camera)

        if i == 0:
            _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
                calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data, camera[None]._data, 1, 0)
            total_fore_hist, total_back_hist = \
                model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image, 
                                                    foreground_distance, background_distance, True)

        data = {
            'image': img[None].cuda(),
            'camera': camera[None].cuda(),
            'body2view_pose': init_pose[None].cuda(),
            'closest_template_views': closest_template_views[None].cuda(),
            'closest_orientations_in_body': closest_orientations_in_body[None].cuda(),
            'fore_hist': total_fore_hist.cuda(),
            'back_hist': total_back_hist.cuda()
        }
        pred = model._forward(data, visualize=False, tracking=True)

        if cfg.output_video:
            pred['optimizing_result_imgs'] = []
            model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
            video.write(cv2.resize(pred['optimizing_result_imgs'][0][0], cfg.output_size))
        
        init_pose = pred['opt_body2view_pose'][-1][0].cpu()
        index = get_closest_template_view_index(init_pose, orientations)
        closest_template_view = template_views[index*num_sample_contour_points:(index+1)*num_sample_contour_points, :]
        _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
            calculate_basic_line_data(closest_template_view[None], init_pose[None]._data, camera[None]._data, 1, 0)
        fore_hist, back_hist = \
            model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image, 
                                                foreground_distance, background_distance, True)
        total_fore_hist = (1 - fore_learn_rate) * total_fore_hist + fore_learn_rate * fore_hist
        total_back_hist = (1 - back_learn_rate) * total_back_hist + back_learn_rate * back_hist

    video.release()