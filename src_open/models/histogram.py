import torch
from torch import nn
from .base_model import BaseModel
from .deep_ac import calculate_basic_line_data

class SingleSoftHistograms(BaseModel):
    eps = 1e-7

    default_conf = {
        'num_bin': 32,
        'hmin': 0.0,
        'hmax': 1.0,
        'num_channel': 3,
        'unconsidered_line_length': 1,
        'considered_line_length': 18,
        
        # 'differentiable': True
    }

    # def __init__(self, unconsidered_line_length, considered_line_length,
    #             num_channel, num_bin, hmin=0, hmax=1):
    def _init(self, conf):
        self.unconsidered_line_length = conf.unconsidered_line_length
        self.considered_line_length = conf.considered_line_length

        self.hmin = conf.hmin
        self.hmax = conf.hmax
        self.num_bin = conf.num_bin
        self.delta = float(conf.hmax - conf.hmin) / float(conf.num_bin)
        self.num_channel = conf.num_channel
        self.total_hist_size = 1
        for _ in range(self.num_channel):
            self.total_hist_size *= self.num_bin
    
    # image: (b, c, h, w), tensor
    def get_segmentation_from_hist(self, image, fore_hist, back_hist):
        assert image.shape[1] == self.num_channel
        assert torch.max(image) <= self.hmax and torch.min(image) >= self.hmin
        batch_size = image.shape[0]
        width = image.shape[3]
        height = image.shape[2]

        image_pf, image_pb = self.get_pf_pb_from_hist(image, fore_hist, back_hist)
        # image_bin_index = self.image_value_to_bin_index(image)  # (b, h, w)
        # image_bin_index = image_bin_index.view(batch_size, -1)  # (b, h*w)
        # image_pf = torch.gather(fore_hist, index=image_bin_index, dim=1)
        # image_pb = torch.gather(back_hist, index=image_bin_index, dim=1)
        
        image_pf += self.eps
        image_pb += self.eps
        # output = (image_pf > image_pb).type(torch.float32) * 255
        output = image_pf / (image_pf + image_pb) * 255
        output = output.view(batch_size, height, width)
        return output
        # cv2.imwrite('./segment_image.png', output[1].type(torch.uint8).cpu().numpy())

    def image_value_to_bin_index(self, image):
        # eps = 1e-7
        # image[(torch.abs(image - self.hmax) < eps)] -= eps
        
        tmp_input_image = torch.round(image * 255).long()
        tmp_input_image = torch.where(tmp_input_image<=255, tmp_input_image, 255)
        bin_index = tmp_input_image // (256 // self.num_bin)

        bin_coefficient = torch.tensor([self.num_channel - i - 1 for i in range(self.num_channel)],
                                        device=image.device)
        bin_coefficient = torch.pow(self.num_bin, bin_coefficient).long()

        for i in range(self.num_channel):
            bin_index[:, i, :, :] *= bin_coefficient[i]

        bin_index = torch.sum(bin_index, dim=1).long()
        bin_index = torch.where(bin_index<self.total_hist_size, bin_index, self.total_hist_size-1)

        return bin_index

    def get_pf_pb_from_hist(self, image, fore_hist, back_hist):
        batch_size = image.shape[0]
        width = image.shape[3]
        height = image.shape[2]
        image_bin_index = self.image_value_to_bin_index(image)  # (b, h, w)
        image_bin_index = image_bin_index.view(batch_size, -1)  # (b, h*w)
        image_pf = torch.gather(fore_hist, index=image_bin_index, dim=1).view(batch_size, height, width)  # + self.eps
        image_pb = torch.gather(back_hist, index=image_bin_index, dim=1).view(batch_size, height, width)  # + self.eps

        return image_pf.view(batch_size, height, width), image_pb.view(batch_size, height, width)

    def histogram_calc(self, input_image, valid_line, normalize: bool=True):
        # assert input_image.shape[1] == self.num_channel
        # assert torch.max(input_image) <= self.hmax and torch.min(input_image) >= self.hmin
        # eps = 1e-7
        # input_image = torch.where(torch.abs(input_image - self.hmax) < eps, input_image-eps, input_image)
        # input_image[(torch.abs(input_image - self.hmax) < eps)] -= eps

        tmp_input_image = torch.round(input_image * 255).long()
        tmp_input_image = torch.where(tmp_input_image<=255, tmp_input_image, 255)
        bin_index = tmp_input_image // (256 // self.num_bin)

        bin_coefficient = torch.tensor([self.num_channel - i - 1 for i in range(self.num_channel)], device=input_image.device)
        bin_coefficient = torch.pow(self.num_bin, bin_coefficient).long()
        for i in range(self.num_channel):
            bin_index[:, i, :, :] *= bin_coefficient[i]
        bin_index = torch.sum(bin_index, dim=1).long()  # (b, h, w)
        bin_index = bin_index.view(input_image.shape[0], -1)
        base_for_scatter = torch.ones_like(bin_index, dtype=torch.float32, device=input_image.device)
        valid = valid_line.reshape(input_image.shape[0], -1)
        bin_index[~valid] = 0
        base_for_scatter[~valid] = 0
        bin_index = torch.where(bin_index<self.total_hist_size, bin_index, 0)
        base_for_scatter = torch.where(bin_index<self.total_hist_size, base_for_scatter, torch.tensor(0, dtype=torch.float32, device=input_image.device))

        hist = torch.zeros((input_image.shape[0], self.total_hist_size), dtype=torch.float32, device=input_image.device)
        hist.scatter_add_(1, bin_index, base_for_scatter)

        if normalize:
            hist = nn.functional.normalize(hist, dim=-1, p=1)
            
        return hist

    def forward(self, image, body2view_pose_data, camera_data, template_view):
        centers_in_body, centers_in_view, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, valid_data_line = \
        calculate_basic_line_data(template_view, body2view_pose_data, camera_data, 1, 0)
        fore_hist, back_hist = self.calculate_histogram(image, centers_in_image, centers_valid, normals_in_image, 
                                                        foreground_distance, background_distance, True)
        # return lines_image[0].permute(1, 2, 0)
        # import cv2
        # cv2.imwrite('lines_image.png', lines_image[0].permute(1, 2, 0).flip(2).mul(255).byte().detach().cpu().numpy())
        # import ipdb
        # ipdb.set_trace()
        # seg_image = self.get_segmentation_from_hist(image, fore_hist, back_hist)
        # seg_image = self.get_segmentation_from_hist(lines_image, fore_hist, back_hist)
        # import cv2
        # cv2.imwrite('seg_image.png', seg_image[0].detach().cpu().numpy())
        # import ipdb
        # ipdb.set_trace()
        return fore_hist, back_hist # , seg_image[0]

    def calculate_histogram(self, image: torch.Tensor, centers_in_image: torch.Tensor, centers_valid: torch.Tensor, normals_in_image: torch.Tensor, 
                            foreground_distance: torch.Tensor, background_distance: torch.Tensor, noramlize: bool):

        device = centers_in_image.device
        batch_size = centers_in_image.shape[0]
        n_correspondence_line = centers_in_image.shape[1]
        height = image.shape[2]
        width = image.shape[3]
        n_channel = image.shape[1]
        # assert n_channel == num_channel

        # tmp_considered_line_length = torch.zeros_like(foreground_distance, dtype=torch.float32, device=device).unsqueeze(-1)
        # fore_line_length = torch.cat((foreground_distance.unsqueeze(-1),
        #                                   tmp_considered_line_length), dim=-1)
        # back_line_length = torch.cat((background_distance.unsqueeze(-1),
        #                                   tmp_considered_line_length), dim=-1)
        # fore_line_length[..., -1] = considered_line_length
        # back_line_length[..., -1] = considered_line_length
        # fore_line_length, _ = torch.min(fore_line_length, dim=-1)
        # back_line_length, _ = torch.min(back_line_length, dim=-1)
        fore_line_length = foreground_distance.clamp(max=float(self.considered_line_length))
        back_line_length = background_distance.clamp(max=float(self.considered_line_length))

        centers = centers_in_image
        normals = normals_in_image

        interpolate_step = torch.arange(-self.considered_line_length, self.considered_line_length + 1, device=device) \
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, n_correspondence_line, -1, 1)
        normals = normals.unsqueeze(-2).expand(-1, -1, interpolate_step.shape[-2], -1)
        interpolate_normals = interpolate_step * normals
        centers = centers.unsqueeze(-2).expand(-1, -1, interpolate_step.shape[-2], -1)
        points_in_correspondence_lines = centers + interpolate_normals  # (b, n_correspondence_line, line_length, 2)
        points_in_correspondence_lines = torch.round(points_in_correspondence_lines)
        points_valid1 = torch.logical_and(points_in_correspondence_lines[..., 0] >= 0, points_in_correspondence_lines[..., 0] < width)
        points_valid2 = torch.logical_and(points_in_correspondence_lines[..., 1] >= 0, points_in_correspondence_lines[..., 1] < height)
        points_valid = torch.logical_and(points_valid1, points_valid2)
        # points_valid = (points_in_correspondence_lines[..., 0] >= 0) & \
        #                 (points_in_correspondence_lines[..., 0] < width) & \
        #                 (points_in_correspondence_lines[..., 1] >= 0) & \
        #                 (points_in_correspondence_lines[..., 1] < height)
        
        points_in_correspondence_lines_x = (points_in_correspondence_lines[..., 0] / width) * 2 - 1
        points_in_correspondence_lines_y = (points_in_correspondence_lines[..., 1] / height) * 2 - 1
        points = torch.cat((points_in_correspondence_lines_x[..., None], points_in_correspondence_lines_y[..., None]), dim=-1)
        lines_image = torch.nn.functional.grid_sample(image, points, mode='nearest', align_corners=False)
        
        # valid_fore_line = interpolate_step.squeeze(-1) <= fore_line_length.unsqueeze(-1)
        # valid_back_line = interpolate_step.squeeze(-1) >= -back_line_length.unsqueeze(-1)
        # valid_line = torch.logical_and(valid_fore_line, valid_back_line).unsqueeze(1).expand(-1, n_channel, -1, -1)
        # centers_valid = centers_valid.unsqueeze(1).unsqueeze(-1).expand(-1, n_channel, -1, valid_line.shape[3])
        # valid_line = torch.logical_and(valid_line, centers_valid)
        # valid_line = torch.logical_and(valid_line, points_valid[:, None])
        # # valid_line = valid_line & centers_valid & points_valid.unsqueeze(1)
        # # unconsidered_valid_line = interpolate_step.squeeze(-1) >= self.unconsidered_line_length
        # unconsidered_valid_line = interpolate_step.abs().squeeze(-1) >= self.unconsidered_line_length
        # valid_line = torch.logical_and(valid_line, unconsidered_valid_line[:, None])
        # # valid_line[..., -unconsidered_line_length + considered_line_length + 1
        # #                 :unconsidered_line_length + considered_line_length] = False
        # lines_image = lines_image + (1 - valid_line.float()) * self.hmax * 10

        valid_fore_line = interpolate_step.squeeze(-1) <= fore_line_length.unsqueeze(-1)
        valid_back_line = interpolate_step.squeeze(-1) >= -back_line_length.unsqueeze(-1)
        valid_line = torch.logical_and(valid_fore_line, valid_back_line)
        centers_valid = centers_valid.unsqueeze(-1).expand(-1, -1, valid_line.shape[-1])
        valid_line = torch.logical_and(valid_line, centers_valid)
        valid_line = torch.logical_and(valid_line, points_valid)
        unconsidered_valid_line = interpolate_step.abs().squeeze(-1) >= self.unconsidered_line_length
        valid_line = torch.logical_and(valid_line, unconsidered_valid_line)

        # return lines_image

        fore_hist = self.histogram_calc(lines_image[:, :, :, :self.considered_line_length].clone(),
                                        valid_line[:, :, :self.considered_line_length], normalize=noramlize)
        back_hist = self.histogram_calc(lines_image[:, :, :, self.considered_line_length + 1:].clone(),
                                        valid_line[:, :, self.considered_line_length + 1:], normalize=noramlize)

        return fore_hist, back_hist

    def calculate(self, image, data_lines, noramlize):
        centers_in_image = data_lines['centers_in_image']
        centers_valid = data_lines['centers_valid']
        normals_in_image = data_lines['normals_in_image']
        foreground_distance = data_lines['foreground_distance']
        background_distance = data_lines['background_distance']
        fore_hist, back_hist = self.calculate_histogram(image, centers_in_image, centers_valid, normals_in_image,
                                                        foreground_distance, background_distance, noramlize)
        # import ipdb
        # ipdb.set_trace()
        return fore_hist, back_hist

    def _forward(self, data):
        raise NotImplementedError

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError