import os
import inspect
import torch
from torch import nn
from .geometry.wrappers import Pose, Camera

def get_file_list(dir, file_list, ext=None):
    new_dir = dir
    if os.path.isfile(dir):
        if ext is None:
            file_list.append(dir)
        else:
            if ext in dir.split('.')[-1]:
                file_list.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir, s)
            get_file_list(new_dir, file_list, ext)

    return file_list

def get_class(mod_name, base_path, base_dir, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
       the module named mod_name, child of base_path.
    """
    file_list = []
    mod_path = None
    get_file_list(base_dir, file_list, 'py')
    for file in file_list:
        file_name = os.path.basename(file).split('.')[0]
        if file_name == mod_name and '__' not in file:
            whole_path = file[:-3].replace('/', '.')
            p = whole_path.find(base_path)
            mod_path = whole_path[p:]
            break

    if mod_path == None:
        raise NotImplementedError

    # mod_path = '{}.{}'.format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[''])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]

# offset_angle: (n) tensor
# offset_translation: (n) tensor
def generate_random_aa_and_t(min_offset_angle, max_offset_angle, min_offset_translation, max_offset_translation):
    if isinstance(min_offset_angle, float):
        min_offset_angle = torch.tensor([min_offset_angle], dtype=torch.float32)
    if isinstance(max_offset_angle, float):
        max_offset_angle = torch.tensor([max_offset_angle], dtype=torch.float32)
    if isinstance(min_offset_translation, float):
        min_offset_translation = torch.tensor([min_offset_translation], dtype=torch.float32)
    if isinstance(max_offset_translation, float):
        max_offset_translation = torch.tensor([max_offset_translation], dtype=torch.float32)

    n = min_offset_angle.shape[0]
    axis = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    angle = (torch.rand(n) * (max_offset_angle - min_offset_angle) + min_offset_angle).unsqueeze(-1) / 180 * 3.1415926
    aa = axis * angle

    direction = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    t = (torch.rand(n) * (max_offset_translation - min_offset_translation) + min_offset_translation).unsqueeze(-1)
    trans = direction * t

    return aa, trans

# p2d: (n, 2) or (b, n, 2)
# return: (4) or (b, 4), [center_x, center_y, w, h]
def get_bbox_from_p2d(p2d):

    bbox_min, _ = torch.min(p2d, dim=-2)
    bbox_max, _ = torch.max(p2d, dim=-2)

    bbox_center = (bbox_min + bbox_max) / 2
    bbox_wh = bbox_max - bbox_min
    bbox = torch.cat((bbox_center, bbox_wh), dim=-1)
    return bbox

def vertex_on_normal_to_image(centers, normals, step):
    return centers + normals * step

def get_closest_template_view_index(body2view_pose: Pose, orientations_in_body):
    orientation = body2view_pose.R.inverse() @ body2view_pose.t.unsqueeze(-1)
    orientation = torch.nn.functional.normalize(orientation, dim=-2).transpose(-1, -2)
    _, index = torch.max(torch.sum(orientation * orientations_in_body, dim=-1), dim=-1)

    return index

def get_closest_k_template_view_index(body2view_pose: Pose, orientations_in_body, k):
    orientation = body2view_pose.R.inverse() @ body2view_pose.t.unsqueeze(-1)
    orientation = torch.nn.functional.normalize(orientation, dim=-2).transpose(-1, -2)
    _, indices = torch.topk(torch.sum(orientation * orientations_in_body, dim=-1), k=k, dim=-1)
    return indices

def project_correspondences_line(template_view, body2view_pose: Pose, camera: Camera, num_sample_center=None):
    if num_sample_center != None:
        step = template_view.shape[1] // num_sample_center
        sample_template_view = template_view[:, ::step, :]
    else:
        sample_template_view = template_view
    centers_in_body = sample_template_view[..., :3]
    normals_in_body = sample_template_view[..., 3:6]
    foreground_distance = sample_template_view[..., 6]
    background_distance = sample_template_view[..., 7]

    centers_in_view = body2view_pose.transform(centers_in_body)
    centers_in_image, centers_valid = camera.view2image(centers_in_view)
    normals_in_view = body2view_pose.rotate(normals_in_body)
    normals_in_image = torch.nn.functional.normalize(normals_in_view[..., :2], dim=-1)

    cur_foreground_distance = foreground_distance * camera.f[..., 0].unsqueeze(-1) / centers_in_view[..., 2]
    cur_background_distance = background_distance * camera.f[..., 0].unsqueeze(-1) / centers_in_view[..., 2]

    data_lines = {'centers_in_body': centers_in_body,
                 'centers_in_view': centers_in_view,
                 'centers_in_image': centers_in_image,
                 'centers_valid': centers_valid,
                 'normals_in_image': normals_in_image,
                 'foreground_distance': cur_foreground_distance,
                 'background_distance': cur_background_distance}

    if torch.any(torch.isnan(data_lines['normals_in_image'])) or torch.any(torch.isnan(data_lines['centers_in_image'])) \
            or torch.any(torch.isnan(data_lines['centers_in_body'])) or torch.any(torch.isnan(data_lines['centers_in_view'])):
            import ipdb;
            ipdb.set_trace();

    return data_lines

def get_lines_image(change_template_view, image, closest_template_views, closest_orientations_in_body,
                    body2view_pose, camera, normal_line_length, num_sample_center=None, mode='nearest'):
    height, width = image.shape[2:]
    if change_template_view:
        index = get_closest_template_view_index(body2view_pose, closest_orientations_in_body)
        template_view = torch.stack([closest_template_views[b][index[b]]
                                     for b in range(closest_template_views.shape[0])])
    else:
        template_view = torch.stack([closest_template_views[b][0]
                                     for b in range(closest_template_views.shape[0])])
    data_lines = project_correspondences_line(template_view, body2view_pose, camera, num_sample_center)
    centers_in_image = data_lines['centers_in_image']
    normals_in_image = data_lines['normals_in_image']
    interpolate_step = torch.arange(-normal_line_length, normal_line_length, device=image.device).unsqueeze(0).unsqueeze(0)\
                           .unsqueeze(-1).expand(centers_in_image.shape[0], centers_in_image.shape[1], -1, -1) + 0.5
    centers = centers_in_image.unsqueeze(2).expand(-1, -1, interpolate_step.shape[2], -1)
    normals = normals_in_image.unsqueeze(2).expand(-1, -1, interpolate_step.shape[2], -1)
    points = centers + interpolate_step * normals
    points[..., 0] = (points[..., 0] / width) * 2 - 1
    points[..., 1] = (points[..., 1] / height) * 2 - 1
    lines_image = torch.nn.functional.grid_sample(image, points, mode=mode, align_corners=False)
    return lines_image, data_lines, template_view

def masked_mean(x, mask, dim, confindence=None):
    mask = mask.float()
    if confindence is not None:
        mask *= confindence
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def checkpointed(cls, do=True):
    '''Adapted from the DISK implementation of Michał Tyszkiewicz.'''
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(
                        super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls

def pack_lr_parameters(params, base_lr, lr_scaling, logger):
    '''Pack each group of parameters with the respective scaled learning rate.
    '''
    from collections import defaultdict

    filters, scales = tuple(zip(*[
        (n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info('Parameters with scaled learning rate:\n{}'.format(
                {s: [n for n, _ in ps] for s, ps in scale2params.items()
                 if s != 1}))
    lr_params = [{'lr': scale*base_lr, 'params': [p for _, p in ps]}
                 for scale, ps in scale2params.items()]
    return lr_params