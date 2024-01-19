import os
import torch
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
import cv2
import copy
import warnings

from ..utils.geometry.wrappers import Pose
from ..models import get_model
from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
# from ..models.tracker_deploy1 import project_correspondences_line, calculate_basic_line_data
from torch import nn

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.
    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model

class CropAndResizeImage(nn.Module):
    def __init__(self, resize, pad_size, crop_border):
        super(CropAndResizeImage, self).__init__()
        self.resize = resize
        self.pad_size = pad_size
        self.crop_border = crop_border

    def forward(self, image, bbox, camera_data_input):
        _, height, width, _ = image.shape
        x1, x2, y1, y2 = bbox
        x1 = (x1-self.crop_border).clamp(min=0, max=width-1)
        x2 = (x2+self.crop_border).clamp(min=0, max=width-1)
        y1 = (y1-self.crop_border).clamp(min=0, max=height-1)
        y2 = (y2+self.crop_border).clamp(min=0, max=height-1)
        img = image[:, int(y1):int(y2+1), int(x1):int(x2+1), :3]
        img = img.permute(0, 3, 1, 2)
        camera_data = camera_data_input
        # camera_data[]

        _, _, h, w = img.shape
        scale = self.resize / max(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        img = torch.nn.functional.interpolate(img, size=(h_new, w_new), mode='bilinear')
        img_padded = torch.zeros((1, 3, self.pad_size, self.pad_size), dtype=torch.float)
        img_padded[:, :, :h_new, :w_new] = img
        img_padded = img_padded / 255

        # imgs_scale = []
        # for s in range(self.num_scale):
        #     image_scale = s ** 2
        #     hh = self.pad_size / image_scale
        #     ww = self.pad_size / image_scale
        #     img_scale = torch.nn.functional.interpolate(img_padded, size=(hh, ww), mode='bilinear')
        #     imgs_scale.append(img_scale)
        
        import ipdb
        ipdb.set_trace()
 

def main(cfg):

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'deploy_cfg.yml')
    assert ('load_cfg' in cfg)
    # assert ('load_model' in cfg)
    assert (Path(cfg.load_cfg).exists())
    # assert (Path(cfg.load_model).exists())
    train_cfg = OmegaConf.load(cfg.load_cfg)
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
    model.eval().cpu()

    histogram_model = model.histogram
    extractor_model = model.extractor
    contour_feature_model = model.contour_feature_map_extractor
    boundary_predictor_model = model.boundary_predictor
    derivative_calculator_model = model.derivative_calculator

    deploy_units = ct.ComputeUnit.CPU_ONLY

    # ori_image = torch.zeros((1, 1440, 1920, 4), dtype=torch.float)
    # ori_camera_data = torch.tensor([1920, 1440, 500, 600, 960, 720], dtype=torch.float)
    # bbox = torch.tensor([200, 400, 200, 500], dtype=torch.float)
    # crop_resize_model = CropAndResizeImage(train_cfg.data.resize, train_cfg.data.pad, train_cfg.data.crop_border)
    # crop_resize_model(ori_image, bbox)

    deploy_input = np.load('data/deploy_input.npz')
    image_input = torch.from_numpy(deploy_input['image'])
    pose_data_input = torch.from_numpy(deploy_input['pose'])
    camera_data_input = torch.from_numpy(deploy_input['camera'])
    template_views = torch.from_numpy(deploy_input['template_views'])
    template_view = template_views[:, 0]
    image_inputs = []
    camera_data_inputs = []
    for i in range(3):
        h, w = image_input.shape[2:]
        image_scale = 2 ** (2-i)
        image_inputs.append(torch.nn.functional.interpolate(image_input, size=(h // int(image_scale), w // int(image_scale))))
        camera_data_inputs.append(camera_data_input / image_scale)

    inp = (image_input, pose_data_input, camera_data_input, template_view)
    jit_histogram_model = torch.jit.trace(histogram_model, example_inputs=inp).eval()
    fore_hist, back_hist = jit_histogram_model(image_input, pose_data_input, camera_data_input, template_view)

    extractor_model = reparameterize_model(extractor_model)
    jit_extractor_model = torch.jit.trace(extractor_model, example_inputs=image_input).eval()
    feature_inputs = jit_extractor_model(image_input)

    feature_inputs = list(feature_inputs)
    # template_view_n = template_view[:, None]
    jit_contour_feature_models = []
    pose_data_inputs = [pose_data_input, pose_data_input, pose_data_input]
    for i in range(3):
        # image_inputs[i] = image_inputs[i][:, None]
        # feature_inputs[i] = feature_inputs[i][:, None]
        # pose_data_inputs[i] = pose_data_inputs[i][:, None]
        # camera_data_inputs[i] = camera_data_inputs[i][:, None]
        inp = (image_inputs[i], feature_inputs[i], pose_data_inputs[i], camera_data_inputs[i], template_view, fore_hist, back_hist)
        jit_contour_feature_models.append(torch.jit.trace(contour_feature_model, example_inputs=inp).eval())
    normals_in_image, centers_in_image, centers_in_body, \
    lines_image_pf_segments, lines_image_pb_segments, valid_data_line, lines_amplitude, lines_slop, lines_feature = \
        jit_contour_feature_models[0](image_inputs[0], feature_inputs[0], pose_data_inputs[0],
                                      camera_data_inputs[0], template_view, fore_hist, back_hist)
    
    inp = (lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude)
    jit_boundary_predictor_model = torch.jit.trace(boundary_predictor_model, example_inputs=inp).eval()

    # distributions, distribution_mean, distribution_variance, distribution_standard_deviation = \
    distributions, distribution_mean, distribution_uncertainties = \
        jit_boundary_predictor_model(lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude)
    # distribution_uncertainties = 1 / distribution_variance
    
    inp = (normals_in_image, centers_in_image, centers_in_body, pose_data_inputs[0], camera_data_inputs[0], 
           valid_data_line, distributions, distribution_mean, distribution_uncertainties)
    jit_derivative_calculator_model = torch.jit.trace(derivative_calculator_model, example_inputs=inp).eval()

    image_input = ct.TensorType(name='image', shape=image_input.shape, dtype=np.float32)
    pose_data_input = ct.TensorType(name='pose_data', shape=pose_data_input.shape, dtype=np.float32)
    camera_data_input = ct.TensorType(name='camera_data', shape=camera_data_input.shape, dtype=np.float32)
    template_view = ct.TensorType(name='template_view', shape=template_view.shape, dtype=np.float32)
    histogram_mlmodel = ct.convert(jit_histogram_model, inputs=[image_input, pose_data_input, camera_data_input, template_view],
                                   minimum_deployment_target=ct.target.iOS16,
                                   outputs=[ct.TensorType(name='fore_hist'), ct.TensorType(name='back_hist')], # ct.TensorType(name='seg_image')],
                                   # outputs=[ct.TensorType(name='lines_image')],
                                   compute_precision=ct.precision.FLOAT16,
                                   # compute_precision=ct.precision.FLOAT16,
                                   compute_units = deploy_units,
                                   # compute_units=ct.ComputeUnit.CPU_ONLY
                                   # compute_units=ct.ComputeUnit.CPU_AND_GPU
                                   # compute_units=ct.ComputeUnit.ALL
                                   )
    histogram_mlmodel.save(os.path.join(logger.log_dir, "histogram.mlpackage"))
    
    extractor_mlmodel = ct.convert(jit_extractor_model, inputs=[image_input],
                                   outputs=[ct.TensorType(name='feature0'), ct.TensorType(name='feature1'), ct.TensorType(name='feature2')],
                                   minimum_deployment_target=ct.target.iOS16,
                                   # compute_precision=ct.precision.FLOAT32,
                                   compute_precision=ct.precision.FLOAT16,
                                   compute_units = deploy_units,
                                 # compute_units=ct.ComputeUnit.CPU_ONLY
                                 # compute_units=ct.ComputeUnit.CPU_AND_GPU
                                   # compute_units=ct.ComputeUnit.ALL
                                   )
    extractor_mlmodel.save(os.path.join(logger.log_dir, "extractor.mlpackage"))

    fore_hist = ct.TensorType(name='fore_hist', shape=fore_hist.shape, dtype=np.float32)
    back_hist = ct.TensorType(name='back_hist', shape=back_hist.shape, dtype=np.float32)
    jit_contour_feature_mlmodels = []
    for i in range(3):
        image_input = ct.TensorType(name='image', shape=image_inputs[i].shape, dtype=np.float32)
        feature_input = ct.TensorType(name='feature', shape=feature_inputs[i].shape, dtype=np.float32)
        # pose_data_input = ct.TensorType(name='pose_data_input', shape=pose_data_inputs[i].shape, dtype=np.float32)
        # camera_data_input = ct.TensorType(name='camera_data_input', shape=camera_data_inputs[i].shape, dtype=np.float32)
        jit_contour_feature_mlmodels.append(ct.convert(jit_contour_feature_models[i], 
                                                       inputs=[image_input, feature_input, pose_data_input, camera_data_input,
                                                               template_view, fore_hist, back_hist],
                                                        minimum_deployment_target=ct.target.iOS16,
                                                        outputs=[ct.TensorType(name='normals_in_image'), ct.TensorType(name='centers_in_image'), 
                                                                 ct.TensorType(name='centers_in_body'), ct.TensorType(name='lines_image_pf_segments'),
                                                                 ct.TensorType(name='lines_image_pb_segments'), ct.TensorType(name='valid_data_line'), 
                                                                 ct.TensorType(name='lines_amplitude'), ct.TensorType(name='lines_slop'), 
                                                                 ct.TensorType(name='lines_feature')],
                                                        # outputs=[ct.TensorType(name='normals_in_image'), ct.TensorType(name='centers_in_image'), 
                                                        #          ct.TensorType(name='centers_in_body'), ct.TensorType(name='valid_data_line'), 
                                                        #          ct.TensorType(name='lines_feature')],
                                                        # compute_precision=ct.precision.FLOAT32,
                                                        compute_precision=ct.precision.FLOAT16,
                                                        compute_units = deploy_units,
                                                        # compute_units=ct.ComputeUnit.CPU_ONLY
                                                        # compute_units=ct.ComputeUnit.CPU_AND_GPU
                                                        # compute_units=ct.ComputeUnit.ALL
                                                        ))
        jit_contour_feature_mlmodels[i].save(os.path.join(logger.log_dir, f"contour_feature_extractor{i}.mlpackage"))

    lines_feature = ct.TensorType(name='lines_feature', shape=lines_feature.shape, dtype=np.float32)
    lines_image_pf_segments = ct.TensorType(name='lines_image_pf_segments', shape=lines_image_pf_segments.shape, dtype=np.float32)
    lines_image_pb_segments = ct.TensorType(name='lines_image_pb_segments', shape=lines_image_pb_segments.shape, dtype=np.float32) 
    lines_slop = ct.TensorType(name='lines_slop', shape=lines_slop.shape, dtype=np.float32)
    lines_amplitude = ct.TensorType(name='lines_amplitude', shape=lines_amplitude.shape, dtype=np.float32)
    boundary_predictor_mlmodel = ct.convert(jit_boundary_predictor_model, 
                                            inputs=[lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude],
                                            # inputs=[lines_feature],
                                            outputs=[ct.TensorType(name='distributions'),
                                                     ct.TensorType(name='distribution_mean'),
                                                     ct.TensorType(name='distribution_uncertainties')],
                                                     # ct.TensorType(name='distribution_variance'), 
                                                     # ct.TensorType(name='distribution_standard_deviation')],
                                            minimum_deployment_target=ct.target.iOS16,
                                            # compute_precision=ct.precision.FLOAT32,
                                            compute_precision=ct.precision.FLOAT16,
                                            compute_units = deploy_units,
                                            # compute_units=ct.ComputeUnit.CPU_ONLY
                                            # compute_units=ct.ComputeUnit.CPU_AND_GPU
                                            # compute_units=ct.ComputeUnit.ALL
                                            )
    boundary_predictor_mlmodel.save(os.path.join(logger.log_dir, "boundary_predictor.mlpackage"))

    normals_in_image = ct.TensorType(name='normals_in_image', shape=normals_in_image.shape, dtype=np.float32)
    centers_in_image = ct.TensorType(name='centers_in_image', shape=centers_in_image.shape, dtype=np.float32)
    centers_in_body = ct.TensorType(name='centers_in_body', shape=centers_in_body.shape, dtype=np.float32)
    valid_data_line = ct.TensorType(name='valid_data_line', shape=valid_data_line.shape, dtype=np.float32)
    distributions = ct.TensorType(name='distributions', shape=distributions.shape, dtype=np.float32)
    distribution_mean = ct.TensorType(name='distribution_mean', shape=distribution_mean.shape, dtype=np.float32)
    distribution_uncertainties = ct.TensorType(name='distribution_uncertainties', shape=distribution_uncertainties.shape, dtype=np.float32)
    derivative_calculator_mlmodel = ct.convert(jit_derivative_calculator_model, 
                                            inputs=[normals_in_image, centers_in_image, centers_in_body, pose_data_input, camera_data_input, 
                                                    valid_data_line, distributions, distribution_mean, distribution_uncertainties],
                                            outputs=[ct.TensorType(name='gradient'), ct.TensorType(name='hessian')],
                                            minimum_deployment_target=ct.target.iOS16,
                                            compute_precision=ct.precision.FLOAT32,
                                            compute_units = deploy_units,
                                            # compute_precision=ct.precision.FLOAT16,
                                            # compute_units=ct.ComputeUnit.CPU_ONLY
                                            # compute_units=ct.ComputeUnit.CPU_AND_GPU
                                            # compute_units=ct.ComputeUnit.ALL
                                            )
    derivative_calculator_mlmodel.save(os.path.join(logger.log_dir, "derivative_calculator.mlpackage"))

    # import ipdb
    # ipdb.set_trace()

    # image_inputs = [
    #     torch.rand(1, 3, cfg.input_size[0], cfg.input_size[1], dtype=torch.float32),
    #     torch.rand(1, 3, cfg.input_size[0] // 2, cfg.input_size[1] // 2, dtype=torch.float32),
    #     torch.rand(1, 3, cfg.input_size[0] // 4, cfg.input_size[1] // 4, dtype=torch.float32)
    #     # torch.zeros(1, 3, cfg.input_size[0], cfg.input_size[1], dtype=torch.float32),
    #     # torch.zeros(1, 3, cfg.input_size[0] // 2, cfg.input_size[1] // 2, dtype=torch.float32),
    #     # torch.zeros(1, 3, cfg.input_size[0] // 4, cfg.input_size[1] // 4, dtype=torch.float32)
    # ]
    # feature_inputs = [
    #     torch.rand(1, 16, cfg.input_size[0], cfg.input_size[1], dtype=torch.float32),
    #     torch.rand(1, 16, cfg.input_size[0] // 2, cfg.input_size[1] // 2, dtype=torch.float32),
    #     torch.rand(1, 16, cfg.input_size[0] // 4, cfg.input_size[1] // 4, dtype=torch.float32)
    #     # torch.zeros(1, 16, cfg.input_size[0], cfg.input_size[1], dtype=torch.float32),
    #     # torch.zeros(1, 16, cfg.input_size[0] // 2, cfg.input_size[1] // 2, dtype=torch.float32),
    #     # torch.zeros(1, 16, cfg.input_size[0] // 4, cfg.input_size[1] // 4, dtype=torch.float32)
    # ]
    # # image_input = torch.zeros(
    # #     1, 3, cfg.input_size[0], cfg.input_size[1], dtype=torch.float32
    # # )  # Batch size = 1
    # # feature_input = torch.zeros(
    # #     1, 16, cfg.input_size[0], cfg.input_size[1], dtype=torch.float32
    # # )  # Batch size = 1
    # R = torch.eye(3, dtype=torch.float32)[None].reshape(1, 9)
    # t = torch.zeros(1, 3, dtype=torch.float32)
    # pose_data_input = torch.cat((R, t), dim=-1)
    # # pose_data_input = torch.eye(3, dtype=torch.float32)[None]
    # camera_data_input = torch.tensor([cfg.input_size[0], cfg.input_size[1], 160, 160, 160, 160], dtype=torch.float32)[None]
    # template_view = torch.ones(1, 200, 8, dtype=torch.float32)
    # fore_hist = torch.rand(1, 32768, dtype=torch.float32)
    # back_hist = torch.rand(1, 32768, dtype=torch.float32)
    # centers_in_image = torch.zeros(1, 200, 2, dtype=torch.float32) 
    # centers_valid = torch.ones(1, 200, dtype=torch.bool) 
    # normals_in_image = torch.zeros(1, 200, 2, dtype=torch.float32) 
    # foreground_distance = torch.ones(1, 200, dtype=torch.float32)
    # background_distance = torch.ones(1, 200, dtype=torch.float32)

    # inp = (image_inputs[0], pose_data_input, camera_data_input, template_view)
    # jit_histogram_model = torch.jit.trace(histogram_model, example_inputs=inp).eval()

    # extractor_model = reparameterize_model(extractor_model)
    # jit_extractor_model = torch.jit.trace(extractor_model, example_inputs=image_inputs[0]).eval()

    # run_iter_model = reparameterize_model(model)
    # jit_run_iter_models = []
    # for i in range(3):
    #     inp = (image_inputs[i], feature_inputs[i], pose_data_input, camera_data_input, template_view, fore_hist, back_hist)
    #     jit_run_iter_models.append(torch.jit.trace(run_iter_model, example_inputs=inp).eval())
    # # inp = (image_input, feature_input, pose_data_input, camera_data_input, template_view, fore_hist, back_hist)
    # # jit_model = torch.jit.trace(model, example_inputs=inp).eval()

    # precision = ct.precision.FLOAT32
    # target = ct.target.iOS16

    # ios_image_input = ct.TensorType(name='image', shape=image_inputs[0].shape, dtype=np.float32)
    # ios_pose_data_input = ct.TensorType(name='pose_data', shape=pose_data_input.shape, dtype=np.float32)
    # ios_camera_data_input = ct.TensorType(name='camera_data', shape=camera_data_input.shape, dtype=np.float32)
    # ios_template_view = ct.TensorType(name='template_view', shape=template_view.shape, dtype=np.float32)
    # ios_fore_hist = ct.TensorType(name='fore_hist', shape=fore_hist.shape, dtype=np.float32)
    # ios_back_hist = ct.TensorType(name='back_hist', shape=back_hist.shape, dtype=np.float32)
    # histogram_mlmodel = ct.convert(jit_histogram_model, inputs=[ios_image_input, ios_pose_data_input, ios_camera_data_input, ios_template_view],
    #                                outputs=[ct.TensorType(name='fore_hist'), ct.TensorType(name='back_hist')], # ct.TensorType(name='seg_image')],
    #                                # outputs=[ct.TensorType(name='lines_image')],
    #                                minimum_deployment_target=target,
    #                                compute_precision=precision,
    #                                # compute_precision=ct.precision.FLOAT16,
    #                              # compute_units=ct.ComputeUnit.CPU_ONLY
    #                              # compute_units=ct.ComputeUnit.CPU_AND_GPU
    #                                compute_units=ct.ComputeUnit.ALL)
    # histogram_mlmodel.save(os.path.join(logger.log_dir, "histogram.mlpackage"))

    # extractor_mlmodel = ct.convert(jit_extractor_model, inputs=[ios_image_input],
    #                                outputs=[ct.TensorType(name='feature0'), ct.TensorType(name='feature1'), ct.TensorType(name='feature2')],
    #                                minimum_deployment_target=target,
    #                                compute_precision=precision,
    #                              # compute_units=ct.ComputeUnit.CPU_ONLY
    #                              # compute_units=ct.ComputeUnit.CPU_AND_GPU
    #                                compute_units=ct.ComputeUnit.ALL)
    # extractor_mlmodel.save(os.path.join(logger.log_dir, "extractor.mlpackage"))

    # run_iter_mlmodels = []
    # for i in range(3):
    #     ios_image_input = ct.TensorType(name='image', shape=image_inputs[i].shape, dtype=np.float32)
    #     ios_feature_input = ct.TensorType(name='feature', shape=feature_inputs[i].shape, dtype=np.float32)
    #     run_iter_mlmodels.append(ct.convert(jit_run_iter_models[i],
    #                                        inputs=[ios_image_input, ios_feature_input, ios_pose_data_input, ios_camera_data_input, 
    #                                        ios_template_view, ios_fore_hist, ios_back_hist],
    #                                        outputs=[ct.TensorType(name='gradient'), ct.TensorType(name='hessian'),
    #                                        ct.TensorType(name='distribution_mean'), ct.TensorType(name='distribution_uncertainties')],
    #                                        minimum_deployment_target=target,
    #                                        compute_precision=precision,
    #                                      # compute_precision=ct.precision.FLOAT16,
    #                                      # compute_units=ct.ComputeUnit.CPU_ONLY
    #                                      # compute_units=ct.ComputeUnit.CPU_AND_GPU
    #                                        compute_units=ct.ComputeUnit.ALL))
    #     run_iter_mlmodels[i].save(os.path.join(logger.log_dir, f"run_iter{3-(i+1)}.mlpackage"))

    # # import ipdb
    # # ipdb.set_trace()