import math
import os

import torch
import cv2
import os.path as osp
import json
import numpy as np
import ray

from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from ..models import get_model
from ..dataset import get_dataset
import warnings
from ..utils.tensor import batch_to_device
from ..utils.utils import get_closest_template_view_index, get_closest_k_template_view_index
from ..utils.geometry.wrappers import Pose
# from pyro.distributions import Uniform
from ..models.deep_ac import calculate_basic_line_data


def generate_samples(mc_samples, pose_opt, batch_size, dtype, device):
    pi = 3.1415926
    trans_low = torch.empty((batch_size, 3), dtype=dtype, device=device)
    trans_high = torch.empty((batch_size, 3), dtype=dtype, device=device)
    trans_low[..., 0] = -1
    trans_high[..., 0] = 1
    trans_low[..., 1] = 0
    trans_high[..., 1] = 1
    trans_low[..., 2] = 0.000
    trans_high[..., 2] = 0.025
    trans_sampler = Uniform(trans_low, trans_high)
        
    rot_low = torch.empty((batch_size, 3), dtype=dtype, device=device)
    rot_high = torch.empty((batch_size, 3), dtype=dtype, device=device)
    rot_low[..., 0] = -1
    rot_high[..., 0] = 1
    rot_low[..., 1] = 0
    rot_high[..., 1] = 1
    rot_low[..., 2] = 0 / 180 * pi
    rot_high[..., 2] = 25 / 180 * pi
    rot_sampler = Uniform(rot_low, rot_high)

    tmp_trans_samples = trans_sampler.rsample((mc_samples,))
    tmp_rot_samples = rot_sampler.rsample((mc_samples,))

    offset_trans_samples = torch.empty(tmp_trans_samples.shape, dtype=dtype, device=device)
    offset_trans_samples[..., 0] = tmp_trans_samples[..., 2] * torch.sqrt(1 - tmp_trans_samples[..., 0] ** 2) * torch.cos(2*pi*tmp_trans_samples[..., 1])
    offset_trans_samples[..., 1] = tmp_trans_samples[..., 2] * torch.sqrt(1 - tmp_trans_samples[..., 0] ** 2) * torch.sin(2*pi*tmp_trans_samples[..., 1])
    offset_trans_samples[..., 2] = tmp_trans_samples[..., 2] * tmp_trans_samples[..., 0]
    offset_rot_samples = torch.empty(tmp_rot_samples.shape, dtype=dtype, device=device)
    offset_rot_samples[..., 0] = tmp_rot_samples[..., 2] * torch.sqrt(1 - tmp_rot_samples[..., 0] ** 2) * torch.cos(2*pi*tmp_rot_samples[..., 1])
    offset_rot_samples[..., 1] = tmp_rot_samples[..., 2] * torch.sqrt(1 - tmp_rot_samples[..., 0] ** 2) * torch.sin(2*pi*tmp_rot_samples[..., 1])
    offset_rot_samples[..., 2] = tmp_rot_samples[..., 2] * tmp_rot_samples[..., 0]
    pose_offset = Pose.from_aa(offset_rot_samples, offset_trans_samples)

    pose_samples = pose_opt @ pose_offset
    return pose_samples

def write_json(anno_path, instance):
    if not anno_path.exists():
        anno_path.parent.mkdir(exist_ok=True)
    with open(str(anno_path), 'w') as f:
        json.dump(instance, f)

@torch.no_grad()
def tracking(cfg, train_cfg, data_conf, logger, device='cuda'):

    # Load model
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

    # Create dataset
    dataset = get_dataset(data_conf.name)(data_conf)
    data_loader = dataset.get_data_loader('test')

    def log_result(log_msg, result, num):
        tmp_result = result.copy()
        for key in tmp_result.keys():
            tmp_result[key] /= num
        for key, value in tmp_result.items():
            log_msg += "{}:{:.4f}| ".format(key, value)
        logger.info(log_msg)
        # print(log_msg)
        return tmp_result

    def AUC_log_result(log_msg, result, num, obj_name):
        tmp_result = result.copy()
        for key in tmp_result.keys():
            tmp_result[key] /= num
        # accumulate
        area_under_curve = 0.0
        for key, value in tmp_result.items():
            area_under_curve += value             
        area_under_curve = 100.0 * area_under_curve * 0.2 / len(tmp_result)

        log_msg = "AUC:{}/{:.4f}| ".format(obj_name, area_under_curve)
        logger.info(log_msg)
        # print(log_msg)

        auc_result = {}
        auc_result['area_under_curve'] = area_under_curve
        return auc_result

    test_results = {}
    total_result = {
        '5cm_5d': 0,
        '5cm': 0,
        '5d': 0,
        '2cm_2d': 0,
        '2cm': 0,
        '2d': 0,
        'ADD_0.1d': 0,
        'ADD_0.05d': 0,
        'ADD_0.02d': 0,
        'num': 0
    }

    total_num = 0

    if data_conf.name == 'OPT' and 'soda' in data_conf.test_obj_names:
        model.optimizer.tikhonov_matrix[0, 0] = 500000
        model.optimizer.tikhonov_matrix[1, 1] = 500000
        model.optimizer.tikhonov_matrix[2, 2] = 500000
    fore_learn_rate = cfg.fore_learn_rate
    back_learn_rate = cfg.back_learn_rate

    output_poses = []

    for it, ori_data in enumerate(data_loader):
        ori_data = batch_to_device(ori_data, device, non_blocking=True)
        obj_name = ori_data['obj_name'][0]
        slice_name = ori_data['slice_name'][0]

        if ori_data['img_id'] == ori_data['last_img_id']:
            body2view_pose = ori_data['gt_body2view_pose']
            num = 0
            result = {
                '5cm_5d': 0,
                '5cm': 0,
                '5d': 0,
                '2cm_2d': 0,
                '2cm': 0,
                '2d': 0,
                'ADD_0.1d': 0,
                'ADD_0.05d': 0,
                'ADD_0.02d': 0,
                'num': 0
            }
            if cfg.output_video:
                video = cv2.VideoWriter(osp.join(logger.log_dir, obj_name + "_" + slice_name + ".avi"),  # 
                                        cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, cfg.output_size)

            if cfg.output_pose:
                output_poses = []

        if ori_data['OPT'].item() and ori_data['obj_start'].item():
            lim = [1, 100]
            rng = np.arange(lim[0], lim[1] + 1)
            obj_ADD_result = {thr: 0 for thr in rng}
            obj_total_num = 0

        # get closest template view
        # index = get_closest_template_view_index(body2view_pose, ori_data['orientations_in_body'])
        # closest_template_views = torch.stack([ori_data['template_views']
        #                                       [:, ind * ori_data['n_sample']:(ind + 1) * ori_data['n_sample'], :]
        #                                       for ind in index])
        # closest_orientations_in_body = ori_data['orientations_in_body'][:, index]

        if 'random_multi_sample' in cfg and cfg['random_multi_sample']:
            pose_samples = generate_samples(50, body2view_pose, body2view_pose.shape[0], body2view_pose.dtype, body2view_pose.device)
            pose_samples[0, 0] = body2view_pose
            body2view_pose = pose_samples[:, 0]

        skip_template_view = data_conf.skip_template_view
        indices = get_closest_k_template_view_index(body2view_pose,
                                                    ori_data['orientations_in_body'],
                                                    data_conf.get_top_k_template_views * skip_template_view)
        # closest_template_views = torch.stack([ori_data['template_views']
        #                                       [0, ind * ori_data['n_sample']:(ind + 1) * ori_data['n_sample'], :]
        #                                       for ind in indices[0, ::skip_template_view]])[None]
        # closest_orientations_in_body = ori_data['orientations_in_body'][:, indices[0, ::skip_template_view]]
        closest_template_views = []
        closest_orientations_in_body = []
        for i, _ in enumerate(indices):
            closest_template_views.append(torch.stack([ori_data['template_views']
                                                      [0, ind * ori_data['n_sample']:(ind + 1) * ori_data['n_sample'], :]
                                                      for ind in indices[i, ::skip_template_view]]))
            closest_orientations_in_body.append(ori_data['orientations_in_body'][0, indices[i, ::skip_template_view]])
        closest_template_views = torch.stack(closest_template_views)
        closest_orientations_in_body = torch.stack(closest_orientations_in_body)

        expand_size = closest_template_views.shape[0]
        ori_data['image'] = ori_data['image'].expand(expand_size, -1, -1, -1)
        # init histogram
        if ori_data['img_id'] == ori_data['last_img_id']:
            _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
            calculate_basic_line_data(closest_template_views[:, 0], body2view_pose._data, ori_data['camera']._data, 1, 0)
            total_fore_hist, total_back_hist = \
                model.histogram.calculate_histogram(ori_data['image'], centers_in_image, centers_valid, normals_in_image, 
                                                    foreground_distance, background_distance, True)
            # seg_img = region_based_constraint.histogram.get_segmentation_from_hist(ori_data['image'], fore_hist, back_hist) \
            #     .detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite('test.png', seg_img[0])
        else:
            total_fore_hist = total_fore_hist.expand(expand_size, -1)
            total_back_hist = total_back_hist.expand(expand_size, -1)

        data = {
            'image': ori_data['image'],
            # 'mask_visib': mask_visib,
            # 'edge_visib': edge_visib,
            # 'edge': edge,
            'camera': ori_data['camera'],
            'body2view_pose': body2view_pose,
            'aligned_vertex': ori_data['aligned_vertex'].expand(expand_size, -1, -1),
            'gt_body2view_pose': ori_data['gt_body2view_pose'],
            'closest_template_views': closest_template_views,
            'closest_orientations_in_body': closest_orientations_in_body,
            'diameter': ori_data['diameter'].expand(expand_size),
            'image_path': ori_data['image_path'] * expand_size,
            'obj_name': ori_data['obj_name'] * expand_size,
            'output_name': ori_data['output_name'] * expand_size,
            'fore_hist': total_fore_hist,
            'back_hist': total_back_hist,
            'sysmetric': ori_data['sysmetric'].expand(expand_size),
            'random_multi_sample': cfg['random_multi_sample'] if 'random_multi_sample' in cfg else False, 
        }

        # if ori_data['img_id'] == ori_data['last_img_id']:
        #     import ipdb
        #     ipdb.set_trace()

        pred, losses, metrics = model.forward_eval(data, visualize=False, tracking=True)
        index = 0
        # _, index = pred['d_distribution_mean'][-1].abs().mean(1).min(0)
        # if ori_data['img_id'] == ori_data['last_img_id']:
        #     index = 0
        pred['opt_body2view_pose'][-1] = pred['opt_body2view_pose'][-1][index, None]
        closest_template_views = closest_template_views[index, None]
        ori_data['image'] = ori_data['image'][index, None]
        total_fore_hist = total_fore_hist[index, None]
        total_back_hist = total_back_hist[index, None]
        for key, value in metrics.items():
            metrics[key] = value[index, None]

        if cfg.output_video:
            pred['optimizing_result_imgs'] = []
            model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
            video.write(cv2.resize(pred['optimizing_result_imgs'][0][0], cfg.output_size))

            # output_seg_img_path = os.path.join(logger.log_dir, pred['output_name'][0] + '_seg.png')
            # seg_img = pred['seg_imgs'][0]
            # cv2.imwrite(output_seg_img_path, seg_img)
            # for i in range(len(pred['weight_imgs'])):
            #     for j, weight_img in enumerate(pred['weight_imgs'][i]):
            #         output_name = data['output_name'][j]
            #         output_weight_result_path = os.path.join(logger.log_dir, output_name + '_' + str(i) + '_weight.png')
            #         cv2.imwrite(output_weight_result_path, weight_img)
            # import ipdb;
            # ipdb.set_trace();

        success = (metrics['R_error'] < 5) and (metrics['t_error'] < 0.05)
        # success = (metrics['R_error'] < 20) and (metrics['t_error'] < 0.2)
        # success = True

        if cfg.output_pose:
            output_R = pred['opt_body2view_pose'][-1].R.cpu().view(-1)
            output_t = pred['opt_body2view_pose'][-1].t.cpu().view(-1)
            if success:
                if (metrics['R_error'] < 5) and (metrics['t_error'] < 0.05):
                    output_poses.append(torch.cat((torch.ones((1))*2, output_R, output_t)))
                else:
                    output_poses.append(torch.cat((torch.ones((1)), output_R, output_t)))
            else:
                output_poses.append(torch.cat((torch.zeros((1)), output_R, output_t)))

        if success or ori_data['OPT'].item():
            body2view_pose = pred['opt_body2view_pose'][-1]
        else:
            body2view_pose = ori_data['gt_body2view_pose']

        # update histogram
        index = get_closest_template_view_index(body2view_pose, ori_data['orientations_in_body'])[0]
        template_view = ori_data['template_views'][:, index * ori_data['n_sample']:(index + 1) * ori_data['n_sample'], :]
        _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
            calculate_basic_line_data(template_view, body2view_pose._data, ori_data['camera']._data, 1, 0)
        fore_hist, back_hist = \
                model.histogram.calculate_histogram(ori_data['image'], centers_in_image, centers_valid, normals_in_image, 
                                                    foreground_distance, background_distance, True)
        # _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
        #     calculate_basic_line_data(closest_template_views[:, 0], body2view_pose._data, ori_data['camera']._data, 1, 0)
        # fore_hist, back_hist = \
        #         model.histogram.calculate_histogram(ori_data['image'], centers_in_image, centers_valid, normals_in_image, 
        #                                             foreground_distance, background_distance, True)

        if success or ori_data['OPT'].item():
            total_fore_hist = (1 - fore_learn_rate) * total_fore_hist + fore_learn_rate * fore_hist
            total_back_hist = (1 - back_learn_rate) * total_back_hist + back_learn_rate * back_hist
        else:
            total_fore_hist = fore_hist
            total_back_hist = back_hist

        num += 1
        result['5cm_5d'] += ((metrics['R_error'] < 5) and (metrics['t_error'] < 0.05)).float().item()
        result['5cm'] += ((metrics['t_error'] < 0.05)).float().item()
        result['5d'] += ((metrics['R_error'] < 5)).float().item()
        result['2cm_2d'] += ((metrics['R_error'] < 2) and (metrics['t_error'] < 0.02)).float().item()
        result['2cm'] += ((metrics['t_error'] < 0.02)).float().item()
        result['2d'] += ((metrics['R_error'] < 2)).float().item()
        result['ADD_0.1d'] += (metrics['err_add'] < metrics['diameter'] * 0.1).float().item()
        result['ADD_0.05d'] += (metrics['err_add'] < metrics['diameter'] * 0.05).float().item()
        result['ADD_0.02d'] += (metrics['err_add'] < metrics['diameter'] * 0.02).float().item()
        total_result['5cm_5d'] += ((metrics['R_error'] < 5) and (metrics['t_error'] < 0.05)).float().item()
        total_result['5cm'] += ((metrics['t_error'] < 0.05)).float().item()
        total_result['5d'] += ((metrics['R_error'] < 5)).float().item()
        total_result['2cm_2d'] += ((metrics['R_error'] < 2) and (metrics['t_error'] < 0.02)).float().item()
        total_result['2cm'] += ((metrics['t_error'] < 0.02)).float().item()
        total_result['2d'] += ((metrics['R_error'] < 2)).float().item()
        total_result['ADD_0.1d'] += (metrics['err_add'] < metrics['diameter'] * 0.1).float().item()
        total_result['ADD_0.05d'] += (metrics['err_add'] < metrics['diameter'] * 0.05).float().item()
        total_result['ADD_0.02d'] += (metrics['err_add'] < metrics['diameter'] * 0.02).float().item()

        if ori_data['end'].item():
            if cfg.output_video:
                video.release()
            log_msg = "Test|{}/{}| ".format(obj_name, slice_name)
            test_results[obj_name + "_" + slice_name] = log_result(log_msg, result, num)
            test_results[obj_name + "_" + slice_name]['num'] = num

            total_num += num
            log_msg = "Test|Total| "
            log_result(log_msg, total_result, total_num)

            if cfg.output_pose:
                output_poses = torch.stack(output_poses)
                output_poses = output_poses.numpy()
                output_pose_path = os.path.join(logger.log_dir, f"{obj_name}_{slice_name}_pose.txt")
                np.savetxt(output_pose_path, output_poses)
                # import ipdb
                # ipdb.set_trace()
        
        # compute metrics AUC
        if ori_data['OPT'].item():
            obj_total_num += 1 
            for thr in rng:
                obj_ADD_result[thr] += (metrics['err_add'] < metrics['diameter'] * thr * 0.002).float().item()

        if ori_data['OPT'].item() and ori_data['obj_end'].item():
            log_msg = "Test|AUC|{}| ".format(obj_name)
            test_results[obj_name + "_AUC"] = AUC_log_result(log_msg, obj_ADD_result, obj_total_num, obj_name)

        del ori_data, data, pred, losses, metrics#, fore_hist, back_hist

    # log_msg = "Test|Total| "
    # for key in total_result.keys():
    #     total_result[key] /= len(data_loader)
    # for key, value in total_result.items():
    #     log_msg += "{}:{:.4f}| ".format(key, value)
    # logger.info(log_msg)
    log_msg = "Test|Total| "
    test_results['Total'] = log_result(log_msg, total_result, len(data_loader))
    test_results['Total']['num'] = len(data_loader)

    return test_results

@ray.remote(num_cpus=1, num_gpus=0.25)  # release gpu after finishing
def tracking_worker_ray_wrapper(*args, **kwargs): #cfg, logger, train_cfg, data_conf, subset_obj_names, slice, worker_id):
    cfg = OmegaConf.create(args[0])
    train_cfg = OmegaConf.create(args[1])
    data_conf = OmegaConf.create(args[2])
    logger = args[3]
    subset_obj_names = args[4]
    slices = args[5]

    data_conf.test_obj_names = subset_obj_names
    data_conf.test_slices = slices
    print('tracking for ', data_conf.test_obj_names, data_conf.test_slices)
    test_results = tracking(cfg, train_cfg, data_conf, logger)

    return test_results

def test_tracking(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'test_cfg.yml')
    assert ('load_cfg' in cfg)
    assert ('load_model' in cfg)
    assert (Path(cfg.load_cfg).exists())
    assert (Path(cfg.load_model).exists())
    train_cfg = OmegaConf.load(cfg.load_cfg)
    logger.dump_cfg(train_cfg, 'train_cfg.yml')

    assert ('data' in cfg)
    assert ('test_obj_names' in cfg.data)
    assert ('test_slices' in cfg.data)
    data_conf = OmegaConf.merge(train_cfg.data, cfg.data)

    assert ('distribution_length' in cfg)
    train_cfg.models.distribution_length = cfg.distribution_length
    train_cfg.models.line_distribution_extractor.distribution_length = cfg.distribution_length

    test_results_path = Path(logger.log_dir, 'test_results.json')

    if cfg.ray.use_ray ==  False:
        test_results = tracking(cfg, train_cfg, data_conf, logger)
        write_json(test_results_path, test_results)
    else:
        from ..utils.ray_utils import ProgressBar, chunks

        ray.init(num_cpus=math.ceil(cfg.ray.n_obj_workers * cfg.ray.n_slice_workers * cfg.ray.n_cpus_per_worker),
                 num_gpus=math.ceil(cfg.ray.n_obj_workers * cfg.ray.n_slice_workers * cfg.ray.n_gpus_per_worker),
                 local_mode=False, ignore_reinit_error=True)

        obj_names = data_conf.test_obj_names
        slices = data_conf.test_slices
        # data_collections = []
        # for obj_name in obj_names:
        #     for slice in slices:
        #         data_collections.append(tuple(obj_name, slice))
        # all_data_subsets = chunks(data_collections, math.ceil(len(data_collections) / cfg.ray.n_workers))
        # all_obj_subsets = chunks(obj_names, math.ceil(len(obj_names) / cfg.ray.n_workers))
        # all_slice_subsets = chunks(slices, math.ceil(len(obj_names) / cfg.ray.n_slice_workers))
        # print("-------------------cfg: ", OmegaConf.to_container(cfg, resolve=True))
        obj_step = math.ceil(len(obj_names) / cfg.ray.n_obj_workers)
        slice_step = math.ceil(len(slices) / cfg.ray.n_slice_workers)
        # for i in range(0, cfg.ray.n_obj_workers):
        #     obj_sub_names = obj_names[i * obj_step:min((i + 1) * obj_step, len(obj_names))]
        #     for j in range(0, cfg.ray.n_slice_workers):
        #         slice_sub_names = slices[j * slice_step: min((j + 1) * slice_step, len(slices))]
        #         print(obj_sub_names, slice_sub_names)
        tracking_worker_results = [
            tracking_worker_ray_wrapper.remote(
                OmegaConf.to_container(cfg, resolve=True),
                OmegaConf.to_container(train_cfg, resolve=True),
                OmegaConf.to_container(data_conf, resolve=True),
                logger,
                obj_names[i * obj_step:min((i + 1) * obj_step, len(obj_names))],
                slices[j * slice_step: min((j + 1) * slice_step, len(slices))])
            for i in range(0, cfg.ray.n_obj_workers)
            for j in range(0, cfg.ray.n_slice_workers)
        ]
        # tracking_worker_results = [
        #     tracking_worker_ray_wrapper.remote(
        #         OmegaConf.to_container(cfg, resolve=True),
        #         OmegaConf.to_container(train_cfg, resolve=True),
        #         OmegaConf.to_container(data_conf, resolve=True),
        #         logger, obj_subset, slices, worker_id=id)
        #     for id, obj_subset in enumerate(all_obj_subsets)
        # ]
        results = ray.get(tracking_worker_results)
        final_results = {}
        for result in results:
            for key, value in result.items():
                if key != 'Total':
                    final_results[key] = value
                else:
                    if 'Total' not in final_results:
                        final_results['Total'] = value
                    else:
                        num1 = final_results['Total']['num']
                        num2 = value['num']
                        num = num1 + num2
                        for k, v in value.items():
                            if k != 'num':
                                final_results['Total'][k] = (final_results['Total'][k] * num1 + v * num2) / num
                        final_results['Total']['num'] = num
        # print(results)
        # print(final_results)
        write_json(test_results_path, final_results)

def test_refine(cfg):
    from ..trainer.trainer import Trainer
    from pytorch_lightning.callbacks import ProgressBar
    import pytorch_lightning as pl

    assert ('load_cfg' in cfg)
    assert ('load_model' in cfg)
    assert (Path(cfg.load_cfg).exists())
    assert (Path(cfg.load_model).exists())
    assert ('data' in cfg)
    assert ('test_obj_names' in cfg.data)
    assert ('test_num_per_obj' in cfg.data)
    
    train_cfg = OmegaConf.load(cfg.load_cfg)
    cfg = OmegaConf.merge(train_cfg, cfg)

    logger = MyLightningLogger('DeepRBOT', cfg.save_dir)
    logger.dump_cfg(cfg, 'test_cfg.yml')

    logger.info("Setting up data...")
    dataset = get_dataset(cfg.data.name)(cfg.data)
    test_data_loader = dataset.get_data_loader('test')

    logger.info("Creating model...")
    task = Trainer(cfg, test_data_loader)

    # TODO: Load model
    if "load_model" in cfg:
        ckpt = torch.load(cfg.load_model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.load_model))

    model_resume_path = False
    # (
    #     os.path.join(cfg.save_dir, "model_last.ckpt")
    #     if "resume" in cfg
    #     else None
    # )

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.trainer.total_epochs,
        gpus=cfg.device.gpu_ids,
        devices=len(cfg.device.gpu_ids),
        check_val_every_n_epoch=cfg.trainer.val_intervals,
        accelerator="gpu",  # "ddp",
        strategy="ddp",
        log_every_n_steps=cfg.trainer.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[ProgressBar(refresh_rate=0)],  # disable tqdm bar
        # plugins=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        benchmark=True,
        # deterministic=True,
    )

    trainer.test(task, test_data_loader)

def main(cfg):
    globals()['test_'+cfg.task](cfg)