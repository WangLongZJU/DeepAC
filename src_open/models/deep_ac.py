
import torch
import torch.nn as nn
import numpy as np
import cv2
import math

from .base_model import BaseModel
from ..models import get_model
from ..utils.utils import masked_mean, get_closest_template_view_index # , project_correspondences_line
from ..utils.geometry.losses import scaled_barron, error_add, error_add_s

# @torch.jit.script
def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M

@torch.jit.script
def transform_p3d(body2view_pose_data, p3d):
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    t = body2view_pose_data[..., 9:]
    return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2)

@torch.jit.script
def rotate_p3d(body2view_pose_data, p3d):
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    return p3d @ R.transpose(-1, -2)

@torch.jit.script
def project_p3d(camera_data, p3d):
    eps=1e-4

    z = p3d[..., -1]
    valid1 = z > eps
    z = z.clamp(min=eps)
    p2d = p3d[..., :-1] / z.unsqueeze(-1)

    f = camera_data[..., 2:4]
    c = camera_data[..., 4:6]
    p2d = p2d * f.unsqueeze(-2) + c.unsqueeze(-2)

    size = camera_data[..., :2]
    size = size.unsqueeze(-2)
    # valid2 = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
    valid2 = torch.logical_and(p2d >= 0, p2d <= (size - 1))
    valid2 = torch.logical_and(valid2[..., 0], valid2[..., 1])
    valid = torch.logical_and(valid1, valid2)

    return p2d, valid

@torch.jit.script
def project_correspondences_line(template_view, body2view_pose_data, camera_data):
    sample_template_view = template_view
    centers_in_body = sample_template_view[..., :3]
    normals_in_body = sample_template_view[..., 3:6]
    foreground_distance = sample_template_view[..., 6]
    background_distance = sample_template_view[..., 7]

    # body2view_pose = Pose.from_4x4mat(body2view_pose_data)
    # camera = Camera(camera_data)
    centers_in_view = transform_p3d(body2view_pose_data, centers_in_body)
    # centers_in_view = body2view_pose.transform(centers_in_body)
    centers_in_image, centers_valid = project_p3d(camera_data, centers_in_view)
    # centers_in_image, centers_valid = camera.view2image(centers_in_view)
    # normals_in_view = body2view_pose.rotate(normals_in_body)
    normals_in_view = rotate_p3d(body2view_pose_data, normals_in_body)
    normals_in_image = torch.nn.functional.normalize(normals_in_view[..., :2], dim=-1)

    f = camera_data[..., 2:4]
    cur_foreground_distance = foreground_distance * f[..., 0].unsqueeze(-1) / centers_in_view[..., 2]
    cur_background_distance = background_distance * f[..., 0].unsqueeze(-1) / centers_in_view[..., 2]

    # data_lines = {'centers_in_body': centers_in_body,
    #              'centers_in_view': centers_in_view,
    #              'centers_in_image': centers_in_image,
    #              'centers_valid': centers_valid,
    #              'normals_in_image': normals_in_image,
    #              'foreground_distance': cur_foreground_distance,
    #              'background_distance': cur_background_distance}

    # if torch.any(torch.isnan(data_lines['normals_in_image'])) or torch.any(torch.isnan(data_lines['centers_in_image'])) \
    #         or torch.any(torch.isnan(data_lines['centers_in_body'])) or torch.any(torch.isnan(data_lines['centers_in_view'])):
    #         import ipdb;
    #         ipdb.set_trace();

    return centers_in_body, centers_in_view, centers_in_image, centers_valid, normals_in_image, cur_foreground_distance, cur_background_distance


@torch.jit.script
def calculate_basic_line_data(template_view, body2view_pose_data, camera_data, fscale: float, min_continuous_distance: float):
    # deformed_body2view_pose = Pose.from_4x4mat(body2view_pose_data)
    # camera = Camera(camera_data)
    # calculate basic line data
    centers_in_body, centers_in_view, centers_in_image, centers_valid, normals_in_image, cur_foreground_distance, cur_background_distance \
    = project_correspondences_line(template_view, body2view_pose_data, camera_data)
    foreground_distance = cur_foreground_distance
    background_distance = cur_background_distance
    continuous_distance = torch.cat((foreground_distance.unsqueeze(-1), background_distance.unsqueeze(-1)), dim=-1) / fscale
    continuous_distance, _ = torch.min(continuous_distance, dim=-1)
    
    # calculate valid data line
    valid_data_line = torch.logical_and((continuous_distance >= min_continuous_distance), centers_valid)
 
    # & (data_line['continuous_distance'] > self.line_length_in_segments)

    # if torch.any(torch.isnan(data_lines['normals_in_image'])) or torch.any(torch.isnan(data_lines['centers_in_image'])) \
    #    or torch.any(torch.isnan(data_lines['centers_in_body'])) or torch.any(torch.isnan(data_lines['centers_in_view'])):
    #     import ipdb;
    #     ipdb.set_trace();

    return centers_in_body, centers_in_view, centers_in_image, centers_valid, normals_in_image,\
           foreground_distance, background_distance, valid_data_line

class ContourFeatureMapExtractor(nn.Module):
    
    def __init__(self, histogram, function_amplitude, function_slope, min_continuous_distance, scale, fscale, line_length, 
                 line_length_minus_1, line_length_minus_1_half, line_length_half_minus_1, line_length_in_segments):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()

        self.histogram = histogram
        self.function_amplitude = function_amplitude
        self.function_slope = function_slope
        self.min_continuous_distance = min_continuous_distance
        self.scale = scale
        self.fscale = fscale
        self.line_length = line_length
        self.line_length_minus_1 = line_length_minus_1
        self.line_length_minus_1_half = line_length_minus_1_half
        self.line_length_half_minus_1 = line_length_half_minus_1
        self.line_length_in_segments = line_length_in_segments
        self.eps = 1e-5

    def calculate_segment_probabilities(self, feature, image, normals_in_image, centers_in_image,
                                        fore_hist, back_hist, valid_data_line,
                                        width, height, device):
        normal_step = normals_in_image
        centers = centers_in_image
        interpolate_step = torch.arange(start=0, end=self.line_length, device=device) \
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1) \
            .expand(normal_step.shape[0], normal_step.shape[1], -1, normal_step.shape[2])
        normal_step = normal_step.unsqueeze(2).expand(-1, -1, interpolate_step.shape[2], -1)
        centers = centers.unsqueeze(2).expand(-1, -1, interpolate_step.shape[2], -1)
        interpolate_step = interpolate_step - self.line_length_minus_1_half
        points = centers + interpolate_step * normal_step
        point_start_valid1 = torch.logical_and(points[..., 0, 0] >= 0, points[..., 0, 0] < width)
        point_start_valid2 = torch.logical_and(points[..., 0, 1] >= 0, points[..., 0, 1] < height)
        point_start_valid = torch.logical_and(point_start_valid1, point_start_valid2)
        point_end_valid1 = torch.logical_and(points[..., -1, 0] >= 0, points[..., -1, 0] < width)
        point_end_valid2 = torch.logical_and(points[..., -1, 1] >= 0, points[..., -1, 1] < height)
        point_end_valid = torch.logical_and(point_end_valid1, point_end_valid2)
        # valid_data_line = valid_data_line & point_start_valid & point_end_valid
        valid_data_line = torch.logical_and(valid_data_line, point_start_valid)
        valid_data_line = torch.logical_and(valid_data_line, point_end_valid)

        points_x = (points[..., 0] / width) * 2 - 1
        points_y = (points[..., 1] / height) * 2 - 1
        points = torch.cat((points_x[..., None], points_y[..., None]), dim=-1)

        lines_image = torch.nn.functional.grid_sample(image, points, mode='nearest', align_corners=False)
        lines_image_pf, lines_image_pb = self.histogram.get_pf_pb_from_hist(lines_image, fore_hist, back_hist)
        lines_image_p_zero_index = torch.logical_and((lines_image_pf < self.eps), (lines_image_pb < self.eps))
        lines_image_pf[lines_image_p_zero_index] = self.eps
        lines_image_pb[lines_image_p_zero_index] = self.eps
        lines_image_psum = lines_image_pf + lines_image_pb
        lines_image_pf /= lines_image_psum
        lines_image_pb /= lines_image_psum

        lines_image_pf_segments = torch.zeros(size=(lines_image_pf.shape[0], lines_image_pf.shape[1],
                                                    self.line_length_in_segments), device=device)
        lines_image_pb_segments = torch.zeros(size=(lines_image_pb.shape[0], lines_image_pb.shape[1],
                                                    self.line_length_in_segments), device=device)

        lines_feature = torch.nn.functional.grid_sample(feature, points, mode='bilinear', align_corners=False)
        lines_amplitude = torch.ones(size=(lines_image_pf_segments.shape[0], lines_image_pf_segments.shape[1])).to(device) * self.function_amplitude
        lines_slop = torch.ones(size=(lines_image_pf_segments.shape[0], lines_image_pf_segments.shape[1])).to(device) * self.function_slope

        # for s in range(self.line_length_in_segments):
        #     lines_image_pf_segments[..., s] = torch.prod(lines_image_pf[..., s * scale: (s + 1) * scale], dim=-1)
        #     lines_image_pb_segments[..., s] = torch.prod(lines_image_pb[..., s * scale: (s + 1) * scale], dim=-1)
        lines_image_pf_segments = lines_image_pf
        lines_image_pb_segments = lines_image_pb

        lines_image_pseg_zero_index = torch.logical_and((lines_image_pf_segments < self.eps), (lines_image_pb_segments < self.eps))
        lines_image_pf_segments[lines_image_pseg_zero_index] = self.eps
        lines_image_pb_segments[lines_image_pseg_zero_index] = self.eps
        # lines_image_pf_segments += self.eps
        # lines_image_pb_segments += self.eps
        lines_image_segments_psum = lines_image_pf_segments + lines_image_pb_segments
        lines_image_pf_segments = lines_image_pf_segments / lines_image_segments_psum
        lines_image_pb_segments = lines_image_pb_segments / lines_image_segments_psum

        return lines_image_pf_segments, lines_image_pb_segments, valid_data_line, lines_amplitude, lines_slop, lines_feature

    def forward(self, image, feature, init_body2view_pose_data, camera_data, template_view, fore_hist, back_hist):
        centers_in_body, centers_in_view, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, valid_data_line \
        = calculate_basic_line_data(template_view, init_body2view_pose_data, camera_data, 1, self.min_continuous_distance)

        width, height = feature.shape[2:]
        device = feature.device
        lines_image_pf_segments, lines_image_pb_segments, valid_data_line, lines_amplitude, lines_slop, lines_feature = \
            self.calculate_segment_probabilities(feature, image, normals_in_image, centers_in_image, 
                                                 fore_hist, back_hist, valid_data_line,
                                                 width, height, device)

        return normals_in_image, centers_in_image, centers_in_body, \
               lines_image_pf_segments, lines_image_pb_segments, valid_data_line, lines_amplitude, lines_slop, lines_feature

class BoundaryPredictor(nn.Module):
    def __init__(self, function_slope, function_length, distribution_length, line_distribution_extractor):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.function_slope = function_slope
        self.function_length = function_length
        self.distribution_length = distribution_length
        self.distribution_length_minus_1_half = (distribution_length-1) / 2
        self.line_distribution_extractor = line_distribution_extractor

    def calculate_distribution(self, lines_feature, lines_image_pf_segments, lines_image_pb_segments,
                               lines_slop, lines_amplitude, device):
        function_slop = lines_slop.unsqueeze(-1).expand(-1, -1, self.function_length)
        function_amplitude = lines_amplitude.unsqueeze(-1).expand(-1, -1, self.function_length)
        x = torch.arange(0, self.function_length, dtype=torch.float32, device=device) \
            - float(self.function_length - 1) / 2
        x = x.unsqueeze(0).unsqueeze(0).expand(lines_slop.shape[0], lines_slop.shape[1], -1)
        if self.function_slope == 0:
            function_lookup_f = 0.5 - function_amplitude * torch.sign(x)
        else:
            function_lookup_f = 0.5 - function_amplitude * torch.tanh(x / (2 * function_slop))
        
        function_lookup_b = 1 - function_lookup_f
        min_variance_laplace = (1.0 / (2.0 * torch.pow(torch.atanh(2.0 * function_amplitude[..., 0]), 2.0))).float()
        min_variance_gaussian = function_slop[..., 0]
        min_variance = torch.max(min_variance_laplace, min_variance_gaussian)

        # distributions = torch.zeros(size=(lines_image_pb_segments.shape[0], lines_image_pb_segments.shape[1],
        #                                   self.distribution_length), dtype=torch.float32, device=device)
        statistical_distributions = []
        for d in range(self.distribution_length):
            pf_prod_func = lines_image_pf_segments[..., d:d + self.function_length] * function_lookup_f
            pb_prod_func = lines_image_pb_segments[..., d:d + self.function_length] * function_lookup_b
            statistical_distributions.append(torch.log(pf_prod_func + pb_prod_func).sum(-1).exp())
            # distributions[..., d] = torch.prod(pf_prod_func + pb_prod_func, dim=-1)
        statistical_distributions = torch.stack(statistical_distributions, dim=-1)
        statistical_distributions = torch.nn.functional.normalize(statistical_distributions, dim=-1, p=1)

        # tmp_distributions = distributions
        inp = {'it': 0, 'inner_it': 0, 'lines_feature': lines_feature,
               'distributions': statistical_distributions, 'pf': lines_image_pf_segments}
        # distributions, distribution_uncertainties = self.line_distribution_extractor(inp)
        distributions = self.line_distribution_extractor(inp)
        
        # return distributions
        if torch.any(torch.isnan(distributions)):
            import ipdb;
            ipdb.set_trace();
        
        # if torch.any(torch.isnan(distributions)) or torch.any(torch.isnan(distribution_uncertainties)):
        #     import ipdb;
        #     ipdb.set_trace();

        distribution_mean, distribution_variance, distribution_standard_deviation = \
            self.fit_gaussian_distribution(distributions, min_variance, device)

        return distributions, distribution_mean, 1 / distribution_variance # , distribution_standard_deviation
        # return distribution_mean, distribution_uncertainties # , statistical_distributions

    def fit_gaussian_distribution(self, distributions, min_variance, device):
        # calculate distribution moments
        distribution_step = torch.arange(0, self.distribution_length, device=device).unsqueeze(0).unsqueeze(0) \
            .expand(distributions.shape[0], distributions.shape[1], -1)
        distribution_tmp_mean = torch.sum(distribution_step * distributions, dim=-1).unsqueeze(-1)
        distribution_tmp_variance = torch.sum(
            torch.pow(distribution_step - distribution_tmp_mean, 2) * distributions, dim=-1)
        distribution_mean = (distribution_tmp_mean - self.distribution_length_minus_1_half).squeeze(-1)
        # tmp_min_variance = torch.ones_like(distribution_tmp_variance, dtype=torch.float32, device=device) * min_variance
        # distribution_variance = torch.maximum(distribution_tmp_variance, tmp_min_variance)
        distribution_variance = torch.maximum(distribution_tmp_variance, min_variance)
        distribution_standard_deviation = torch.pow(distribution_variance, 0.5)

        # return distribution_mean, distribution_tmp_variance, distribution_tmp_variance
        return distribution_mean, distribution_variance, distribution_standard_deviation

    def forward(self, lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude):
        return self.calculate_distribution(lines_feature, lines_image_pf_segments, lines_image_pb_segments,
                                           lines_slop, lines_amplitude, lines_feature.device)

class DerivativeCalculator(nn.Module):
    def __init__(self, alternative_optimizing, distribution_length, distribution_length_plus_1_half, learning_rate):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.alternative_optimizing = alternative_optimizing
        self.distribution_length = distribution_length
        self.distribution_length_plus_1_half = distribution_length_plus_1_half
        self.learning_rate = learning_rate
    
    def calculate_gradient_and_hessian(self, it, normals_in_image, centers_in_image, centers_in_body, 
                                       deformed_body2view_pose_data, camera_data, valid_data_line,
                                       distributions, distribution_mean, distribution_uncertainties, device):

        original_normals_in_image = normals_in_image
        original_centers_in_image = centers_in_image
        # centers_in_body = centers_in_body
        centers_in_view = transform_p3d(deformed_body2view_pose_data, centers_in_body)
        # centers_in_view = deformed_body2view_pose.transform(centers_in_body)

        x = centers_in_view[..., 0]
        y = centers_in_view[..., 1]
        z = centers_in_view[..., 2]

        f = camera_data[..., 2:4]
        c = camera_data[..., 4:6]
        fu_z = f[..., 0].unsqueeze(-1) / z
        fv_z = f[..., 1].unsqueeze(-1) / z
        xfu_z = x * fu_z
        yfv_z = y * fv_z
        centers_in_image_u = xfu_z + c[..., 0].unsqueeze(-1)
        centers_in_image_v = yfv_z + c[..., 1].unsqueeze(-1)

        delta_cs = (original_normals_in_image[..., 0]
                    * (centers_in_image_u - original_centers_in_image[..., 0])
                    + original_normals_in_image[..., 1]
                    * (centers_in_image_v - original_centers_in_image[..., 1]))
        
        if self.alternative_optimizing == False:
            dloglikelihood_ddelta_cs = (distribution_mean - delta_cs) * distribution_uncertainties # / distribution_variance
        else:
            if it % 2 == 0:
                dloglikelihood_ddelta_cs = (distribution_mean - delta_cs) * distribution_uncertainties # / distribution_variance
            else:
                dist_idx_upper = (delta_cs + self.distribution_length_plus_1_half).long()
                dist_idx_lower = dist_idx_upper - 1
                # print('min idx lower ', torch.min(dist_idx_lower, dim=-1))
                # print('max idx upper ', torch.max(dist_idx_upper, dim=-1))
                valid_data_line &= (dist_idx_lower >= 0) & (dist_idx_upper < self.distribution_length)
                min_lower = torch.zeros_like(dist_idx_lower, device=device, dtype=torch.long)
                max_upper = torch.zeros_like(dist_idx_upper, device=device, dtype=torch.long)
                max_upper[...] = self.distribution_length - 1
                dist_idx_lower = torch.minimum(torch.maximum(dist_idx_lower, min_lower), max_upper)
                dist_idx_upper = torch.minimum(torch.maximum(dist_idx_upper, min_lower), max_upper)
                # print('refine min idx lower ', torch.min(dist_idx_lower, dim=-1))
                # print('refine max idx upper ', torch.max(dist_idx_upper, dim=-1))
                distribution_upper = torch.gather(input=distributions, index=dist_idx_upper.unsqueeze(-1), dim=-1)
                distribution_lower = torch.gather(input=distributions, index=dist_idx_lower.unsqueeze(-1), dim=-1)
                distribution_min = torch.ones_like(distribution_upper.squeeze(-1)) * 0.00001
                dloglikelihood_ddelta_cs = (torch.log(torch.maximum(distribution_upper.squeeze(-1), distribution_min))
                                            - torch.log(torch.maximum(distribution_lower.squeeze(-1), distribution_min))) \
                                           * self.learning_rate * distribution_uncertainties # / distribution_variance

        ddelta_cs_dcenter_x =  original_normals_in_image[..., 0] * fu_z
        ddelta_cs_dcenter_y =  original_normals_in_image[..., 1] * fv_z
        ddelta_cs_dcenter_z =  (-original_normals_in_image[..., 0] * xfu_z
                                - original_normals_in_image[..., 1] * yfv_z) / z
        ddelta_cs_dcenter = torch.cat((ddelta_cs_dcenter_x.unsqueeze(-1), ddelta_cs_dcenter_y.unsqueeze(-1),
                                       ddelta_cs_dcenter_z.unsqueeze(-1)), dim=-1).unsqueeze(-2)

        R = deformed_body2view_pose_data[..., :9].view(-1, 3, 3)
        deformed_body2view_pose_R = R.unsqueeze(1).expand(-1, ddelta_cs_dcenter.shape[1], -1, -1)

        ddelta_cs_dtranslation = ddelta_cs_dcenter.matmul(deformed_body2view_pose_R)
        ddelta_cs_drotation = ddelta_cs_dtranslation.matmul(-skew_symmetric(centers_in_body))
        ddelta_cs_dtheta = torch.cat((ddelta_cs_drotation, ddelta_cs_dtranslation), dim=-1).squeeze(-2)

        gradients = dloglikelihood_ddelta_cs.unsqueeze(-1) * ddelta_cs_dtheta
        ddelta_cs_dtheta_1 = (ddelta_cs_dtheta * torch.sqrt(distribution_uncertainties).unsqueeze(-1)).unsqueeze(-1)
        # ddelta_cs_dtheta_1 = (ddelta_cs_dtheta / distribution_standard_deviation.unsqueeze(-1)).unsqueeze(-1)
        hessians = ddelta_cs_dtheta_1.matmul(ddelta_cs_dtheta_1.transpose(-1, -2))

        valid_data_line_float1 = valid_data_line.unsqueeze(-1).expand(-1, -1, gradients.shape[-1]).float()
        gradients = gradients * valid_data_line_float1
        valid_data_line_float2 = valid_data_line.unsqueeze(-1).unsqueeze(-1) \
            .expand(-1, -1, hessians.shape[-2], hessians.shape[-1]).float()
        hessians = hessians * valid_data_line_float2

        gradient = gradients.sum(dim=1)
        hessian = hessians.sum(dim=1)

        if torch.any(torch.isnan(gradient)) or torch.any(torch.isnan(hessian)):
            import ipdb;
            ipdb.set_trace();

        return gradient.unsqueeze(-1), hessian

    def forward(self, normals_in_image, centers_in_image, centers_in_body, 
                      deformed_body2view_pose_data, camera_data, valid_data_line,
                      distributions, distribution_mean, distribution_uncertainties, it=0):
        return self.calculate_gradient_and_hessian(it, normals_in_image, centers_in_image, centers_in_body, 
                                                   deformed_body2view_pose_data, camera_data, valid_data_line,
                                                   distributions, distribution_mean, distribution_uncertainties,
                                                   normals_in_image.device)



class DeepAC(BaseModel):
    default_conf = {

        'success_thresh': 5,
        'clamp_error': 50,
        'regulation_distribution_mean': False,
        'down_sample_image_mode': 'bilinear',
        
        'function_length': 8,
        'distribution_length': 16,
        'function_slope': 0.0,
        'function_amplitude': 0.36,
        'min_continuous_distance': 6.0,
        'learning_rate': 1.3,
        'alternative_optimizing': False,

        'debug_check_display': False,
    }

    required_data_keys = {

    }

    strict_conf = False  # need to pass new confs to children models
    eps = 1e-5

    def _init(self, conf):
        self.conf = conf
        self.histogram = get_model(conf.histogram.name)(conf.histogram)
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        self.optimizer = get_model(conf.optimizer.name)(conf.optimizer)

        self.function_length = conf.function_length
        self.distribution_length = conf.distribution_length
        self.function_slope = conf.function_slope
        self.function_amplitude = conf.function_amplitude
        self.min_continuous_distance = conf.min_continuous_distance
        self.learning_rate = conf.learning_rate
        self.precalculate_function_lookup()
        self.precalculate_distribution_variables()
        scale, fscale, line_length, line_length_minus_1, line_length_minus_1_half, line_length_half_minus_1 = \
            self.precalculate_scale_dependent_variables()

        self.contour_feature_map_extractor = \
            ContourFeatureMapExtractor(self.histogram, self.function_amplitude, self.function_slope, 
                                       self.min_continuous_distance, scale, fscale, line_length, line_length_minus_1, 
                                       line_length_minus_1_half, line_length_half_minus_1, self.line_length_in_segments)
        self.line_distribution_extractor = get_model(conf.line_distribution_extractor.name)\
                                                    (conf.line_distribution_extractor)
        self.boundary_predictor =  BoundaryPredictor(self.function_slope, self.function_length, self.distribution_length,
                                                     self.line_distribution_extractor)
        self.derivative_calculator = DerivativeCalculator(conf.alternative_optimizing, self.distribution_length, 
                                                          self.distribution_length_plus_1_half, self.learning_rate)


    def precalculate_function_lookup(self):
        x = torch.arange(0, self.function_length, dtype=torch.float32) - float(self.function_length - 1) / 2
        if self.function_slope == 0:
            self.function_lookup_f = (
                    0.5 - self.function_amplitude * torch.sign(x))  # ((0.0 < x).float() - (x < 0.0).float())
        else:
            self.function_lookup_f = 0.5 - self.function_amplitude * torch.tanh(x / (2 * self.function_slope))
        self.function_lookup_b = 1 - self.function_lookup_f

    def precalculate_distribution_variables(self):
        self.line_length_in_segments = self.function_length + self.distribution_length - 1
        self.distribution_length_minus_1_half = (float(self.distribution_length) - 1) / 2
        self.distribution_length_plus_1_half = (float(self.distribution_length) + 1) / 2
        min_variance_laplace = float(1.0 / (2.0 * pow(math.atanh(2.0 * self.function_amplitude), 2.0)))
        min_variance_gaussian = self.function_slope
        self.min_variance = max(min_variance_laplace, min_variance_gaussian)

    def precalculate_scale_dependent_variables(self):
        scale = 1
        fscale = float(scale)
        line_length = self.line_length_in_segments * scale
        line_length_minus_1 = line_length - 1
        line_length_minus_1_half = float(line_length - 1) * 0.5
        line_length_half_minus_1 = float(line_length) * 0.5 - 1.0

        return scale, fscale, line_length, line_length_minus_1, line_length_minus_1_half, line_length_half_minus_1


    def visualize_optimization(self, body2view_pose, data):
        import cv2
        from ..utils.draw_tutorial import draw_correspondence_lines_in_image

        batch_size = data['image'].shape[0]

        index = get_closest_template_view_index(body2view_pose, data['closest_orientations_in_body'])
        template_view = torch.stack([data['closest_template_views'][b][index[b]]
                                     for b in range(data['closest_template_views'].shape[0])])
        _, _, centers_in_image, centers_valid, normals_in_image, _, _ = \
            project_correspondences_line(template_view, body2view_pose._data, data['camera']._data)

        images = (data['image'].permute(0, 2, 3, 1).detach().cpu().numpy( ) *255).astype(np.uint8).copy()
        centers_in_image = centers_in_image.detach().cpu().numpy()
        centers_valid = centers_valid.detach().cpu().numpy()
        normals_in_image = normals_in_image.detach().cpu().numpy()
        optimizing_result_imgs = None
        for b in range(batch_size):
            optimizing_result_img = draw_correspondence_lines_in_image(images[b], centers_in_image[b],
                                                                       centers_valid[b], normals_in_image[b], 0, center_color=(0, 255, 0))
            optimizing_result_img = cv2.cvtColor(optimizing_result_img, cv2.COLOR_RGB2BGR)
            if optimizing_result_imgs is None:
                optimizing_result_imgs = optimizing_result_img[None]
            else:
                optimizing_result_imgs = np.append(optimizing_result_imgs,  optimizing_result_img[None], axis=0)
            # cv2.imwrite("test.png", display_img)
        data['optimizing_result_imgs'].append(optimizing_result_imgs)
    
    def init_histogram(self, data):
        camera = data['camera']
        gt_body2view_pose = data['gt_body2view_pose']
        image = data['image'] if 'image_aug' not in data else data['image_aug']
        closest_orientations_in_body = data['closest_orientations_in_body']
        closest_template_views = data['closest_template_views']
        index = get_closest_template_view_index(gt_body2view_pose, closest_orientations_in_body)
        template_view = torch.stack([closest_template_views[b][index[b]] for b in range(closest_template_views.shape[0])])

        _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ = \
        calculate_basic_line_data(template_view, gt_body2view_pose._data, camera._data, 1, self.min_continuous_distance)
        fore_hist, back_hist = self.histogram.calculate_histogram(image, centers_in_image, centers_valid, normals_in_image, 
                                                                  foreground_distance, background_distance, True)
        # seg_image = self.histogram.get_segmentation_from_hist(image, fore_hist, back_hist).detach().cpu().numpy().astype(np.uint8)
        # cv2.imwrite('seg_image0.png', seg_image[0])
        # seg_image = self.histogram.get_segmentation_from_hist(data['image'], fore_hist, back_hist).detach().cpu().numpy().astype(np.uint8)
        # cv2.imwrite('seg_image1.png', seg_image[0])
        # import ipdb
        # ipdb.set_trace()
        return fore_hist, back_hist

    # def forward(self, image, feature, init_body2view_pose_data, camera_data, template_view, fore_hist, back_hist, it=0):
    def run_iteration(self, image, feature, init_body2view_pose_data, camera_data, template_view, fore_hist, back_hist, it=0):
        
        if torch.any(torch.isnan(feature)):
            import ipdb;
            ipdb.set_trace();

        normals_in_image, centers_in_image, centers_in_body, \
        lines_image_pf_segments, lines_image_pb_segments, valid_data_line, lines_amplitude, lines_slop, lines_feature = \
            self.contour_feature_map_extractor.forward(image, feature, init_body2view_pose_data, camera_data, template_view, fore_hist, back_hist)

        # distributions, distribution_mean, distribution_variance, distribution_standard_deviation =\
        distributions, distribution_mean, distribution_uncertainties =\
            self.boundary_predictor.forward(lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude)
        # distribution_uncertainties = 1 / distribution_variance

        gradient, hessian = \
        self.derivative_calculator.forward(normals_in_image, centers_in_image, centers_in_body, init_body2view_pose_data, camera_data,
                                           valid_data_line, distributions, distribution_mean, distribution_uncertainties, it)

        # return gradient, hessian, distribution_mean, distribution_uncertainties
        return gradient, hessian


    def _forward(self, data, visualize=False, tracking=False):
        # data['weight_imgs'] = []

        # data['tracking'] = tracking
        # data['visualize'] = visualize

        image = data['image']
        B, _, H, W = image.shape
        features = self.extractor._forward(image)
        
        if tracking == False:
            fore_hist, back_hist = self.init_histogram(data)
        else:
            fore_hist = data['fore_hist']
            back_hist = data['back_hist']

        data['opt_body2view_pose'] = []
        camera = data['camera']
        closest_orientations_in_body = data['closest_orientations_in_body']
        closest_template_views = data['closest_template_views']
        optimizer = self.optimizer
        init_body2view_pose = data['body2view_pose']
        for it, s in enumerate(self.conf.scales):
            image_scale = float(2 ** s)
            camera_pyr = camera.scale(1 / image_scale)
            h_cur = H // int(image_scale)
            w_cur = W // int(image_scale)
            image_pyr = torch.nn.functional.interpolate(image, size=(h_cur, w_cur), mode=self.conf.down_sample_image_mode)
            feature = features[-(s+1)]

            index = get_closest_template_view_index(init_body2view_pose, closest_orientations_in_body)
            template_view = torch.stack([closest_template_views[b][index[b]] for b in range(closest_template_views.shape[0])])
            
            B, A = self.run_iteration(image_pyr, feature, init_body2view_pose._data, camera_pyr._data, template_view, fore_hist, back_hist, it)

            optimizing_pose_q = optimizer(dict(pose=init_body2view_pose, B=B, A=A))
            data['opt_body2view_pose'].append(optimizing_pose_q)
            init_body2view_pose = optimizing_pose_q.detach()

        return data

    def loss(self, pred, data):

        cam = pred['camera']
        def project(body2view_pose, centers_in_body):
            centers_in_view = body2view_pose.transform(centers_in_body)
            return cam.view2image(centers_in_view)

        def reprojection_error(body2view_pose, centers_in_body, gt, valid):
            centers_in_image, _ = project(body2view_pose, centers_in_body)
            err = torch.sum((gt - centers_in_image) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, valid, -1)
            return err

        # def vertices_error(gt_vertex_in_view, body2view_pose, vertex_in_body):
        #     centers_in_view = body2view_pose.transform(vertex_in_body)
        #     err = torch.sum((gt_vertex_in_view - centers_in_view) ** 2, dim=-1)
        #     err = scaled_barron(1., 2.)(err)[0] / 4
        #     err = err.mean(dim=-1)
        #     return err

        losses = {'total': 0.}
        success = None
        gt_body2view_pose = pred['gt_body2view_pose']
        vertex_in_body = pred['aligned_vertex']
        gt_vertex_in_image, gt_vertex_valid = project(gt_body2view_pose, vertex_in_body)
        for i in range(len(pred['opt_body2view_pose'])):
            opt_body2view_pose = pred['opt_body2view_pose'][i]
            # gt_vertex_in_view = gt_body2view_pose.transform(vertex_in_body)
            # ver_error = vertices_error(gt_vertex_in_view, opt_body2view_pose, vertex_in_body)
            # loss_vertex = ver_error / len(pred['opt_body2view_pose'])
            err_reprojection = reprojection_error(opt_body2view_pose, vertex_in_body, gt_vertex_in_image,
                                                  gt_vertex_valid).clamp(max=self.conf.clamp_error)
            loss_reprojection = err_reprojection / len(pred['opt_body2view_pose'])

            if i > 0:
                loss_reprojection = loss_reprojection * success.float()
            scale = len(pred['opt_body2view_pose']) - i
            thresh = self.conf.success_thresh * scale
            success = err_reprojection < thresh

            if self.conf.regulation_distribution_mean:
                opt_distribution_mean = pred['opt_distribution_mean'][i]
                roll_opt_distribution_mean = torch.roll(opt_distribution_mean, 1)
                error_distribution_mean_regulation = (roll_opt_distribution_mean - opt_distribution_mean).abs().mean(-1)
                loss_distribution_mean_regulation = error_distribution_mean_regulation / len(pred['opt_body2view_pose'])
                losses[f'error_distribution_mean_regulation/{i}'] = error_distribution_mean_regulation
                losses['total'] += loss_distribution_mean_regulation

            losses[f'reprojection_error/{i}'] = err_reprojection
            losses['total'] += loss_reprojection

        err_init = reprojection_error(pred['body2view_pose'], vertex_in_body, gt_vertex_in_image, gt_vertex_valid)
        losses['reprojection_error/init'] = err_init

        return losses

    def metrics(self, pred, data):
        metrics = {'diameter': pred['diameter']}  # = self.loss(pred, data)
        vertices = pred['aligned_vertex']
        gt_view2body_pose = pred['gt_body2view_pose'].inv()
        gt_body2view_pose = pred['gt_body2view_pose']
        opt_body2view_pose = pred['opt_body2view_pose'][-1]
        init_body2view_pose = pred['body2view_pose']

        def scaled_pose_error(body2view_pose):
            err_t = torch.norm(body2view_pose.t - gt_body2view_pose.t, dim=-1)
            # err_R = torch.acos((torch.trace(body2view_pose.R.permute(0, 2, 1) @ gt_body2view_pose.R) - 1) / 2)
            err_R = torch.acos((((body2view_pose.R @ gt_view2body_pose.R)
                                .diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1))
            err_R = torch.rad2deg(err_R)
            # err_R, err_t = (body2view_pose @ gt_view2body_pose).magnitude()
            # if self.conf.normalize_dt:
            #     err_t /= torch.norm(gt_view2body_pose.t, dim=-1)
            #
            return err_R, err_t

        with torch.no_grad():
            metrics['R_error'], metrics['t_error'] = scaled_pose_error(opt_body2view_pose)
            metrics['err_add'] = error_add(vertices, gt_body2view_pose, opt_body2view_pose)
            metrics['err_add_s'] = error_add_s(vertices, gt_body2view_pose, opt_body2view_pose)
            metrics['err_add(s)'] = metrics['err_add']
            metrics['err_add(s)'][pred['sysmetric']] = metrics['err_add_s'][pred['sysmetric']]
            init_err_add = error_add(vertices, gt_body2view_pose, init_body2view_pose)
            init_err_add_s = error_add_s(vertices, gt_body2view_pose, init_body2view_pose)
            metrics['err_add(s)_init'] = init_err_add
            metrics['err_add(s)_init'][pred['sysmetric']] = init_err_add_s[pred['sysmetric']]

        return metrics

    def forward_train(self, data):
        pred = self._forward(data)
        losses = self.loss(pred, data)
        metrics = self.metrics(pred, data)

        return pred, losses

    def forward_eval(self, data, visualize, tracking=False):
        pred = self._forward(data, visualize, tracking)
        losses = self.loss(pred, data)
        metrics = self.metrics(pred, data)

        return pred, losses, metrics