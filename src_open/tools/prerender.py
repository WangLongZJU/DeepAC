import os
import os.path as osp
from pathlib import Path
import torch
import imageio
import numpy as np
from tqdm import tqdm
import cv2
import pickle

from ..utils.geometry.body import Body
from ..utils.geometry.render_geometry import GenerateGeodesicPoses, RenderGeometry
from ..utils.geometry.viewer import Viewer
from ..utils.geometry.wrappers import Pose, Camera
from ..utils.draw_tutorial import draw_vertices_to_obj

def preprocess(conf, obj_dirs, obj_paths):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_obj = len(obj_paths)
    sphere_radius = float(conf.sphere_radius)
    n_divide = int(conf.n_divide)
    image_size = torch.tensor([conf.image_size], dtype=torch.int).expand(n_obj).to(device)
    image_border_size = torch.tensor([conf.image_border_size], dtype=torch.int).expand(n_obj).to(device)
    k_min_contour_length = int(conf.k_min_contour_length)
    num_sample_contour_point = int(conf.num_sample_contour_point)
    k_contour_normal_approx_radius = int(conf.k_contour_normal_approx_radius)
    normalize_to_origin = bool(conf.normalize_to_origin)
    output_mask = bool(conf.output_mask)
    output_depth = bool(conf.output_depth)
    output_depth_vertex = bool(conf.output_depth_vertex)
    geometry_unit_in_meter = float(conf.geometry_unit_in_meter)
    maximum_body_diameter = float(conf.maximum_body_diameter)

    body = Body('body_0', obj_paths, geometry_unit_in_meter, maximum_body_diameter,
                normalize_to_origin=normalize_to_origin, device=device)
    ggp = GenerateGeodesicPoses(body.maximum_body_diameter, sphere_radius, image_size,
                                image_border_size, n_divide, device=device)
    view2world_matrix = ggp.view2world_matrix.transpose(0, 1)
    view2world_pose = Pose.from_4x4mat(view2world_matrix[0])
    viewer = Viewer((image_size[0].cpu().item(), image_size[0].cpu().item()),
                    view2world_pose, ggp.virtual_camera, device=device)

    render_geometry = RenderGeometry("render eigen", device=device)
    render_geometry.add_body(body)
    render_geometry.add_viewer(body.name, viewer)
    render_geometry.setup_render_context()

    print('start preprocess: ', obj_paths)
    template_views = 0
    orientations = 0
    for i in tqdm(range(0, view2world_matrix.shape[0])):
        view2world_pose = Pose.from_4x4mat(view2world_matrix[i])
        render_geometry.update_viewer_pose(body.name, view2world_pose)
        depths = render_geometry.render_depth()

        if output_depth_vertex:
            tmp_viewer = render_geometry.viewers[body.name]
            x = torch.arange(0, image_size[0], 1, device=device, dtype=torch.float32)
            y = torch.arange(0, image_size[0], 1, device=device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x, y)
            p2d = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1)\
                .permute(1, 0, 2).reshape(1, -1, 2).expand(n_obj, -1, -1)
            z = depths[body.name].reshape(n_obj, -1)
            p3ds, valids = tmp_viewer.camera.image2view(p2d, z)

        depth = depths[body.name].cpu().numpy()
        for j in range(n_obj):
            obj_path = obj_paths[j]
            output_dir = os.path.join(os.path.dirname(obj_path), 'pre_render')
            if output_mask:
                output_mask_dir = os.path.join(output_dir, 'mask')
                if not os.path.exists(output_mask_dir):
                    os.makedirs(output_mask_dir)
                mask = (depth[j] > 0).astype(np.uint8) * 255
                mask_path = os.path.join(output_mask_dir, str(i).zfill(6) + '.jpg')
                cv2.imwrite(mask_path, mask)
                # tmp_mask_path = os.path.join(output_mask_dir, str(i).zfill(6) + '_tmp.jpg')
                # tmp_mask = (tmp_depth[j].cpu().numpy() > 0).astype(np.uint8) * 255
                # cv2.imwrite(tmp_mask_path, tmp_mask)

            if output_depth:
                output_depth_dir = os.path.join(output_dir, 'depth')
                if not os.path.exists(output_depth_dir):
                    os.makedirs(output_depth_dir)
                depth_path = os.path.join(output_depth_dir, str(i).zfill(6) + '.exr')
                imageio.imwrite(depth_path, depth[j].astype(np.float32))

            if output_depth_vertex:
                tmp_viewer = render_geometry.viewers[body.name]
                tmp_body = render_geometry.bodies[body.name]
                output_depth_vertex_dir = os.path.join(output_dir, 'depth_vertex')
                if not os.path.exists(output_depth_vertex_dir):
                    os.makedirs(output_depth_vertex_dir)
                depth_vertex_path = os.path.join(output_depth_vertex_dir, str(i).zfill(6) + '.obj')
                p3d_in_view = p3ds[j][valids[j]]
                p3d_in_world = tmp_viewer.view2world_pose[j].transform(p3d_in_view)
                p3d_in_body = tmp_body.world2body_pose[j].transform(p3d_in_world)
                p3d_in_body /= geometry_unit_in_meter
                draw_vertices_to_obj(p3d_in_body.cpu().numpy(), depth_vertex_path)

        orientation = (body.world2body_pose @ view2world_pose).R[:, :3, 2].unsqueeze(1).cpu().numpy()
        # orientation = view2world_pose.R[:, :3, 2].unsqueeze(1).cpu().numpy()

        ret, centers_in_body, normals_in_body, foreground_distance, background_distance = \
            render_geometry.generate_point_data(body.name, depths, k_min_contour_length, num_sample_contour_point,
                                                k_contour_normal_approx_radius)

        if not ret:
            import ipdb;
            ipdb.set_trace();

        template_view = np.concatenate((centers_in_body, normals_in_body, np.expand_dims(foreground_distance, axis=-1),
                                        np.expand_dims(background_distance, axis=-1)), axis=-1)

        if i == 0:
            template_views = template_view
            orientations = orientation
        else:
            template_views = np.concatenate((template_views, template_view), axis=1)
            orientations = np.concatenate((orientations, orientation), axis=1)

    for j in range(n_obj):
        obj_path = obj_paths[j]
        obj_name = os.path.basename(obj_path).split('.')[0] + '.pkl'
        output_dir = os.path.join(os.path.dirname(obj_path), 'pre_render')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, obj_name)
        template_view = template_views[j]
        orientation = orientations[j]
        fx = ggp.virtual_camera.f[j, 0].cpu().item()
        fy = ggp.virtual_camera.f[j, 1].cpu().item()
        cx = ggp.virtual_camera.c[j, 0].cpu().item()
        cy = ggp.virtual_camera.c[j, 1].cpu().item()
        head = {'obj_path': obj_path,
                'image_size': (image_size[0].cpu().item(), image_size[0].cpu().item()),
                'num_sample_contour_point': num_sample_contour_point,
                'body_normalize_to_origin': normalize_to_origin,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        if output_mask:
            head['mask_dir'] = os.path.join(output_dir, 'mask')
        if output_depth:
            head['depth_dir'] = os.path.join(output_dir, 'depth')
        dictionary_data = {'head': head, 'template_view': template_view, 'orientation_in_body': orientation}

        with open(output_path, "wb") as pkl_handle:
            pickle.dump(dictionary_data, pkl_handle)

        #with open(output_path, "rb") as pkl_handle:
        #    output = pickle.load(pkl_handle)

    print('finish preprocess: ', obj_paths)

def prerender_RBOT(cfg):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    data_dir = cfg.data_dir
    batch_size = cfg.batch_size
    body_names = list(cfg.body_names)

    if 'all' in body_names:
        body_names = []
        for file_name in os.listdir(data_dir):
            file_path = osp.join(data_dir, file_name)
            if osp.isdir(file_path):
                body_names.append(file_name)

    for i in range(0, len(body_names), batch_size):
        obj_start = i
        obj_end = i + batch_size
        if obj_end > len(body_names):
            obj_end = len(body_names)
        obj_paths = []
        obj_dirs = []
        for body_name in body_names[obj_start:obj_end]:
            obj_path = osp.join(data_dir, body_name, body_name+'.obj')
            if not os.path.exists(obj_path):
                raise FileNotFoundError
            obj_dir = osp.join(data_dir, body_name)
            if not Path(obj_dir).exists():
                Path(obj_dir).mkdir(exist_ok=True)
            new_obj_path = osp.join(obj_dir, body_name + '.obj')
            os.system(f'cp -r "{obj_path}" "{new_obj_path}"')
            obj_path = osp.join(obj_dir, body_name + '.obj')

            obj_dirs.append(obj_dir)
            obj_paths.append(obj_path)

        preprocess(cfg, obj_dirs, obj_paths)


def preprender_BCOT(cfg):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    data_dir = cfg.data_dir
    batch_size = cfg.batch_size
    body_names = list(cfg.body_names)

    if 'all' in body_names:
        body_names = []
        for file_name in os.listdir(data_dir):
            file_path = osp.join(data_dir, file_name)
            if osp.isdir(file_path):
                body_names.append(file_name)

    for i in range(0, len(body_names), batch_size):
        obj_start = i
        obj_end = i + batch_size
        if obj_end > len(body_names):
            obj_end = len(body_names)
        obj_paths = []
        obj_dirs = []
        for body_name in body_names[obj_start:obj_end]:
            obj_path = osp.join(data_dir, body_name+'.obj')
            if not os.path.exists(obj_path):
                raise FileNotFoundError
            obj_dir = osp.join(data_dir, body_name)
            if not Path(obj_dir).exists():
                Path(obj_dir).mkdir(exist_ok=True)
            new_obj_path = osp.join(obj_dir, body_name + '.obj')
            os.system(f'cp -r "{obj_path}" "{new_obj_path}"')
            obj_path = osp.join(obj_dir, body_name + '.obj')

            obj_dirs.append(obj_dir)
            obj_paths.append(obj_path)

        preprocess(cfg, obj_dirs, obj_paths)


# prerender BOP sub datasets, such as HOPE, IC-BIN, ...
# set configs in src/configs/prerender/*.yaml first
def preprender_BOP(cfg):
    data_dir = cfg.data_dir
    batch_size = cfg.batch_size
    body_names = list(cfg.body_names)

    if 'all' in body_names:
        body_names = []
        for file_name in os.listdir(data_dir):
            file_path = osp.join(data_dir, file_name)
            if osp.isdir(file_path):
                body_names.append(file_name)

    for i in range(0, len(body_names), batch_size):
        obj_start = i
        obj_end = i + batch_size
        if obj_end > len(body_names):
            obj_end = len(body_names)
        obj_paths = []
        obj_dirs = []
        for body_name in body_names[obj_start:obj_end]:
            obj_path = osp.join(data_dir, body_name+'.ply')
            if not os.path.exists(obj_path):
                raise FileNotFoundError
            obj_dir = osp.join(data_dir, body_name)
            if not Path(obj_dir).exists():
                Path(obj_dir).mkdir(exist_ok=True)
            new_obj_path = osp.join(obj_dir, body_name + '.ply')
            os.system(f'cp -r "{obj_path}" "{new_obj_path}"')
            obj_path = osp.join(obj_dir, body_name + '.ply')

            obj_dirs.append(obj_dir)
            obj_paths.append(obj_path)

        preprocess(cfg, obj_dirs, obj_paths)


def preprender_OPT(cfg):
    data_dir = cfg.data_dir
    batch_size = cfg.batch_size
    body_names = list(cfg.body_names)

    if 'all' in body_names:
        body_names = []
        for file_name in os.listdir(data_dir):
            file_path = osp.join(data_dir, file_name)
            if osp.isdir(file_path):
                body_names.append(file_name)

    for i in range(0, len(body_names), batch_size):
        obj_start = i
        obj_end = i + batch_size
        if obj_end > len(body_names):
            obj_end = len(body_names)
        obj_paths = []
        obj_dirs = []
        for body_name in body_names[obj_start:obj_end]:
            obj_path = osp.join(data_dir, body_name, body_name+'.obj')
            if not os.path.exists(obj_path):
                raise FileNotFoundError
            obj_dir = osp.join(data_dir, body_name)

            obj_dirs.append(obj_dir)
            obj_paths.append(obj_path)

        preprocess(cfg, obj_dirs, obj_paths)