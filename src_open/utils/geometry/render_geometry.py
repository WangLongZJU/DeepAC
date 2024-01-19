from .body import Body
from .viewer import Viewer
from torch import nn, Tensor
from .wrappers import Camera, Pose
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRasterizer
)
from pytorch3d.io import load_objs_as_meshes
import cv2
import numpy as np
import torch
from pytorch3d.utils import cameras_from_opencv_projection

class IcosahedronItem(object):
    def __init__(self, points: torch.Tensor):
        self.points = points

    def __repr__(self):
        return "Item(%f, %f, %f)" % (self.points[0], self.points[1], self.points[2])

    def __eq__(self, other):
        if isinstance(other, IcosahedronItem):
            ans = ~((other.points[0] < self.points[0]) or
                  ((other.points[0] == self.points[0]) and (other.points[1] < self.points[1])) or
                  ((other.points[0] == self.points[0]) and (other.points[1] == self.points[1]) and (other.points[2] < self.points[2])))
            return ans
        else:
            return False

    def __ne__(self, other):
       return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.__repr__())

class GenerateGeodesicPoses():

    # shape
    # maximum_body_diameter: (n)
    # sphere_radius: (n)
    # image_size: (n)
    # image_border_size: (n)
    def __init__(self, maximum_body_diameter, sphere_radius, image_size, image_border_size, n_divide, device='cpu'):

        n_obj = image_size.shape[0]

        # generate virtual camera intrinsic parameter
        focal_length = (image_size - image_border_size) * sphere_radius / maximum_body_diameter
        principal_point = image_size / 2
        self.virtual_camera = Camera(torch.cat((image_size.unsqueeze(-1), image_size.unsqueeze(-1),
                                                focal_length.unsqueeze(-1), focal_length.unsqueeze(-1),
                                                principal_point.unsqueeze(-1), principal_point.unsqueeze(-1)), dim=1))

        # generate virtual view
        assert 0 < n_divide <= 4
        self.geodesic_points = np.loadtxt('./data/template/geodesic_points_%d.txt' % n_divide)
        self.geodesic_points = torch.from_numpy(self.geodesic_points)\
            .unsqueeze(0).expand(n_obj, -1, -1).to(device).type(torch.float32)

        # x = 0.525731112119133606
        # z = 0.850650808352039932
        # icosahedron_points = torch.tensor([[-x, 0.0, z], [x, 0.0, z], [-x, 0.0, -z], [x, 0.0, -z],
        #                                    [0.0, z, x], [0.0, z, -x], [0.0, -z, x], [0.0, -z, -x],
        #                                    [z, x, 0.0], [-z, x, 0.0], [z, -x, 0.0], [-z, -x, 0.0]],
        #                                   device=device, dtype=torch.float32)
        # icosahedron_ids = torch.tensor([[0, 4, 1], [0, 9, 4], [9, 5, 4], [4, 5, 8], [4, 8, 1],
        #                                 [8, 10, 1], [8, 3, 10], [5, 3, 8], [5, 2, 3], [2, 7, 3],
        #                                 [7, 10, 3], [7, 6, 10], [7, 11, 6], [11, 0, 6], [0, 1, 6],
        #                                 [6, 1, 10], [9, 0, 11], [9, 11, 2], [9, 2, 5], [7, 2, 11]],
        #                                device=device, dtype=torch.int)
        #
        # # self.geodesic_points = set()
        # self.geodesic_points = []
        #
        # for icosahedron_id in icosahedron_ids:
        #     self.subdivide_triangle((icosahedron_points[icosahedron_id[0]]),
        #                             (icosahedron_points[icosahedron_id[1]]),
        #                             (icosahedron_points[icosahedron_id[2]]),
        #                             n_divide)
        #
        # self.geodesic_points = torch.stack(self.geodesic_points).unsqueeze(0).expand(n_obj, -1, -1)

        downwards = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)\
            .unsqueeze(0).unsqueeze(0).expand(n_obj, self.geodesic_points.shape[1], 3)
        view2world_matrix = torch.zeros(size=(n_obj, self.geodesic_points.shape[1], 4, 4),
                                        device=device, dtype=torch.float32)

        view2world_matrix[..., :3, 3] = self.geodesic_points * sphere_radius
        view2world_matrix[..., 3, 3] = 1
        view2world_matrix[..., :3, 2] = -self.geodesic_points
        view2world_matrix[..., :3, 0] = torch.nn.functional.normalize(torch.cross(downwards, -self.geodesic_points), dim=-1)
        view2world_matrix = view2world_matrix.view(-1, 4, 4)
        view2world_matrix[torch.norm(view2world_matrix[..., :3, 0], dim=-1) == 0, :3, 0] = \
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        view2world_matrix = view2world_matrix.view(n_obj, -1, 4, 4)
        view2world_matrix[..., :3, 1] = torch.cross(view2world_matrix[..., :3, 2], view2world_matrix[..., :3, 0])
        # view2world_matrix[0, 121, :, :] = torch.tensor([[0.7269,  0.6251, -0.2844, 0.5050],
        #                                                 [-0.0165, -0.3980, -0.9172, 0.7542],
        #                                                 [-0.6866,  0.6714, -0.2790, 0.3497],
        #                                                 [0, 0, 0, 1]], dtype=torch.float32, device=device)
        self.view2world_matrix = view2world_matrix.clone()
        # draw_vertices_to_obj(view2world.cpu().numpy(), './data/geodesic_points.obj')

        # view2world_matrix[:, :, :3, 3] /= 0.001
        # draw_axis_to_obj(view2world_matrix[0, 122:123].cpu(), './data/camera_pose_template_122.obj')
        # draw_axis_to_obj(view2world_matrix[0, 0:1].cpu(), './data/camera_pose_template_0.obj')
        # draw_axis_to_obj(view2world_matrix[0, 121:122].cpu(), './data/camera_pose.obj')
        # import ipdb;
        # ipdb.set_trace();

    def subdivide_triangle(self, v1, v2, v3, n_divide):
        if n_divide == 0:
            # self.geodesic_points.add(IcosahedronItem(v1))
            # self.geodesic_points.add(IcosahedronItem(v2))
            # self.geodesic_points.add(IcosahedronItem(v3))
            self.geodesic_points.append(v1)
            self.geodesic_points.append(v2)
            self.geodesic_points.append(v3)
        else:
            v12 = torch.nn.functional.normalize(v1+v2, dim=0)
            v13 = torch.nn.functional.normalize(v1+v3, dim=0)
            v23 = torch.nn.functional.normalize(v2+v3, dim=0)
            self.subdivide_triangle(v1, v12, v13, n_divide - 1)
            self.subdivide_triangle(v2, v12, v23, n_divide - 1)
            self.subdivide_triangle(v3, v13, v23, n_divide - 1)
            self.subdivide_triangle(v12, v13, v23, n_divide - 1)

def get_render_Rt_from_extrinsic_Rt(extrinsic_pose: Pose):
    render_R = extrinsic_pose.R
    render_R[:, 0, :] = -render_R[:, 0, :]
    render_R[:, 1, :] = -render_R[:, 1, :]
    render_t = extrinsic_pose.t
    render_t[:, 0] = -render_t[:, 0]
    render_t[:, 1] = -render_t[:, 1]
    return Pose.from_Rt(render_R, render_t)

class RenderGeometry:
    eps = 1e-4
    numeric_max = 1e+7

    def __init__(self, name='none', device='cpu'):
        self.name = name
        self.device = device
        self.bodies = {}
        self.viewers = {}
        self.depth_rasterizers = {}

    def add_body(self, body: Body):
        self.bodies[body.name] = body

    def add_viewer(self, body_name, viewer: Viewer):
        assert self.bodies.get(body_name) is not None
        assert self.bodies.get(body_name).n_body == viewer.num
        self.viewers[body_name] = viewer

    def align_coordinate_and_get_extrinsic_matrix(self, viewer: Viewer, body: Body):
        body2world_pose = body.body2world_pose
        if body.body_coordinate_to_render_coordinate_func is not None:
            body2world_pose = body.body_coordinate_to_render_coordinate_func(body2world_pose)

        world2view_pose = viewer.world2view_pose
        if viewer.view_coordinate_to_render_coordinate_func is not None:
            world2view_pose = viewer.view_coordinate_to_render_coordinate_func(world2view_pose)

        return world2view_pose @ body2world_pose

    def get_render_camera(self, viewer: Viewer, body: Body):

        body2view_pose = viewer.world2view_pose @ body.body2world_pose
        tmp_zeros = torch.zeros_like(viewer.camera.f[..., 0])
        tmp_ones = torch.ones_like(viewer.camera.f[..., 0])
        intrisic_matrix = torch.stack([viewer.camera.f[..., 0], tmp_zeros, viewer.camera.c[..., 0],
                                       tmp_zeros, viewer.camera.f[..., 1], viewer.camera.c[..., 1],
                                       tmp_zeros, tmp_zeros, tmp_ones], dim=-1).reshape(-1, 3, 3).cuda()
        image_size = torch.tensor([viewer.image_size[1], viewer.image_size[0]])\
            .unsqueeze(0).expand(body2view_pose.shape[0], -1).cuda()

        return cameras_from_opencv_projection(body2view_pose.R, body2view_pose.t, intrisic_matrix, image_size).cuda()

        # extrinsic_matrix = self.align_coordinate_and_get_extrinsic_matrix(viewer, body)
        # return PerspectiveCameras(focal_length=viewer.camera.f, principal_point=viewer.camera.c,
        #                           R=extrinsic_matrix.R, T=extrinsic_matrix.t,
        #                           image_size=torch.flip(viewer.camera.size, dims=[1]), in_ndc=False)

    def setup_render_context(self):

        for body_name, body in self.bodies.items():
            if self.viewers.get(body_name) is None:
                print("Skip the body %s when setup render context, because can not find relative viewer!" % body_name)
                continue

            viewer = self.viewers[body_name]
            render_camera = self.get_render_camera(viewer, body)
            render_camera = render_camera.to(self.device)

            # vertex_world = body.meshes.verts_list()[0][0].unsqueeze(0)
            # vertex_camera = render_camera.get_world_to_view_transform().transform_points(vertex_world)
            # vertex_ndc = render_camera.transform_points(vertex_world)
            # vertex_screen = render_camera.transform_points_screen(vertex_world)
            #
            # fx = viewer.camera.f[..., 0] * 2.0 / viewer.camera.size[..., 0]
            # fy = viewer.camera.f[..., 1] * 2.0 / viewer.camera.size[..., 1]
            # px = - (viewer.camera.c[..., 0] - viewer.camera.size[..., 0]/2.0) * 2.0 / viewer.camera.size[..., 0]
            # py = - (viewer.camera.c[..., 1] - viewer.camera.size[..., 1]/2.0) * 2.0 / viewer.camera.size[..., 1]
            # vertex_camera = vertex_camera.squeeze(1)
            # x_ndc = vertex_camera[..., 0] * fx / vertex_camera[..., 2] + px
            # y_ndc = vertex_camera[..., 1] * fy / vertex_camera[..., 2] + py
            # z_ndc = 1 / vertex_camera[..., 2]
            # x_screen = (viewer.camera.size[..., 0] - 1) / 2.0 * (1 - x_ndc)
            # y_screen = (viewer.camera.size[..., 1] - 1) / 2.0 * (1 - y_ndc)

            raster_settings = RasterizationSettings(
                image_size=(viewer.image_size[1], viewer.image_size[0]),#viewer.image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0 # shen
            )

            self.depth_rasterizers[body_name] = \
                MeshRasterizer(cameras=render_camera, raster_settings=raster_settings)

    def update_viewer_pose(self, body_name, view2world_pose: Pose):
        assert self.viewers.get(body_name) is not None
        self.viewers[body_name].view2world_pose = view2world_pose

    def update_body_pose(self, body_name, body2world_pose: Pose):
        assert self.bodies.get(body_name) is not None
        self.bodies[body_name].body2world_pose = body2world_pose

    def render_depth(self):
        depths = {}
        for body_name, depth_rasterizer in self.depth_rasterizers.items():
            body = self.bodies[body_name]
            viewer = self.viewers[body_name]

            body2view_pose = (viewer.world2view_pose @ body.body2world_pose).cuda()
            # tmp_zeros = torch.zeros_like(viewer.camera.f[..., 0])
            # tmp_ones = torch.ones_like(viewer.camera.f[..., 0])
            # intrisic_matrix = torch.stack([viewer.camera.f[..., 0], tmp_zeros, viewer.camera.c[..., 0],
            #                                tmp_zeros, viewer.camera.f[..., 1], viewer.camera.c[..., 1],
            #                                tmp_zeros, tmp_zeros, tmp_ones], dim=-1).reshape(-1, 3, 3).cuda()
            # image_size = torch.tensor([viewer.image_size[1], viewer.image_size[0]]).unsqueeze(0).cuda()
            # tmp_cameras = \
            #     cameras_from_opencv_projection(body2view_pose.R, body2view_pose.t, intrisic_matrix, image_size).cuda()
            # raster_settings = RasterizationSettings(
            #     image_size=(viewer.image_size[1], viewer.image_size[0]),  # viewer.image_size,
            #     blur_radius=0.0,
            #     faces_per_pixel=1,
            #     # bin_size=0
            # )
            # tmp_depth_rasterizers = \
            #     MeshRasterizer(cameras=tmp_cameras, raster_settings=raster_settings)
            # tmp_depth = tmp_depth_rasterizers(body.meshes).zbuf
            # mask = (tmp_depth > 0).type(torch.float32)
            # tmp_depth = tmp_depth * mask

            # extrinsic_matrix = self.align_coordinate_and_get_extrinsic_matrix(viewer, body)
            R_pytorch3d = body2view_pose.R.clone().permute(0, 2, 1)
            T_pytorch3d = body2view_pose.t.clone()
            R_pytorch3d[:, :, :2] *= -1
            T_pytorch3d[:, :2] *= -1
            depth = depth_rasterizer(body.meshes, R=R_pytorch3d, T=T_pytorch3d).zbuf  # (b, h, w, c)
            mask = (depth > 0).type(torch.float32)
            depth = depth * mask

            depths[body_name] = depth

            # depth = depth.cpu().numpy()
            # mask = (depth[0] > 0).astype(np.uint8) * 255
            # cv2.imwrite("mask1.png", mask)
            # mask = (depth[1] > 0).astype(np.uint8) * 255
            # cv2.imwrite("mask2.png", mask)

        return depths

    def generate_valid_contour(self, mask, image_size, k_min_contour_length, k_contour_normal_approx_radius):
        mask_float = mask.type(torch.float32)
        border_sum = torch.sum(mask_float[0, :]) + torch.sum(mask_float[:, 0]) +\
                     torch.sum(mask_float[0, image_size[0]-1]) + torch.sum(mask_float[image_size[1]-1, :])
        # BodyData must fit into image
        assert border_sum.item() == 0

        # cv2.imwrite('mask.png', mask.cpu().numpy().astype(np.uint8)*255)
        contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # image = np.zeros_like(mask.cpu().numpy(), dtype=np.uint8)
        # cv2.drawContours(image, contours, -1, 255)
        # cv2.imwrite('contour.png', image)
        final_contours = 0
        final_normals = 0
        total_contour_length_in_pixel = 0
        i = 0
        for contour in contours:
            contour = contour[:, 0, :]
            if contour.shape[0] > k_min_contour_length:
                if abs(contour[0, 0] - contour[-1, 0]) > 1 or abs(contour[0, 1] - contour[-1, 1]) > 1:
                    print("----- this contour is invalid ! -----")
                    continue
                # calc normal
                contour = torch.from_numpy(contour)
                contour2 = torch.cat((contour, contour), dim=0)
                next_ = contour2[k_contour_normal_approx_radius:k_contour_normal_approx_radius+contour.shape[0]]
                prev_ = contour2[contour.shape[0]-k_contour_normal_approx_radius:-k_contour_normal_approx_radius]
                valid = torch.norm((next_ - prev_).type(torch.float32), dim=-1) > k_contour_normal_approx_radius
                normal_x = -(next_[:, 1] - prev_[:, 1])
                normal_y = (next_[:, 0] - prev_[:, 0])
                normal = torch.cat((normal_x.unsqueeze(-1), normal_y.unsqueeze(-1)), dim=-1)
                normal = torch.nn.functional.normalize(normal.type(torch.float32), dim=-1)
                contour = contour[valid]
                normal = normal[valid]

                if contour.shape[0] < k_min_contour_length:
                    continue

                if i == 0:
                    final_contours = contour
                    final_normals = normal
                else:
                    final_contours = torch.cat((final_contours, contour), dim=0)
                    final_normals = torch.cat((final_normals, normal), dim=0)

                total_contour_length_in_pixel += contour.shape[0]
                i += 1

        if total_contour_length_in_pixel == 0:
            return 0, [], []

        return total_contour_length_in_pixel, final_contours.to(self.device), final_normals.to(self.device)

    def calculate_line_distances(self, image_size, masks, sample_centers, sample_normals, pixel_to_meter):
        n_obj = sample_centers.shape[0]
        n_sample = sample_centers.shape[1]
        width = image_size[0]
        height = image_size[1]
        foreground_distance = np.zeros(shape=(n_obj, n_sample), dtype=np.float)
        background_distance = np.zeros(shape=(n_obj, n_sample), dtype=np.float)

        for i in range(n_obj):
            for j in range(n_sample):
                center_x = sample_centers[i, j, 0]
                center_y = sample_centers[i, j, 1]
                u_out = center_x + 0.5
                v_out = center_y + 0.5
                u_in = center_x + 0.5
                v_in = center_y + 0.5
                normal_x = sample_normals[i, j, 0]
                normal_y = sample_normals[i, j, 1]
                u_step = 0.0
                v_step = 0.0
                if (abs(normal_x) < abs(normal_y)):
                    u_step = normal_x / abs(normal_y)
                    v_step = np.sign(normal_y)
                else:
                    u_step = np.sign(normal_x)
                    v_step = normal_y / abs(normal_x)

                while True:
                    u_in -= u_step
                    v_in -= v_step
                    if not masks[i, int(v_in), int(u_in), 0]:
                        foreground_distance[i, j] = \
                            np.sqrt((u_in - center_x) * (u_in - center_x) +
                                    (v_in - center_y) * (v_in - center_y)) * pixel_to_meter[i, j]
                        break

                while True:
                    u_out += u_step
                    v_out += v_step
                    if int(u_out) < 0 or int(u_out) >= width or int(v_out) < 0 or int(v_out) >= height:
                        background_distance[i, j] = self.numeric_max
                        break
                    if masks[i, int(v_out), int(u_out), 0]:
                        background_distance[i, j] = \
                            np.sqrt((u_out - center_x) * (u_out - center_x) +
                                    (v_out - center_y) * (v_out - center_y)) * pixel_to_meter[i, j]
                        break

        return foreground_distance, background_distance

    def generate_point_data(self, body_name, depths, k_min_contour_length, num_sample_contour_point,
                            k_contour_normal_approx_radius):
        viewer = self.viewers[body_name]
        body = self.bodies[body_name]
        n_obj = viewer.num
        image_size = viewer.image_size
        assert depths[body_name].shape[0] == n_obj
        depth = depths[body_name]
        mask = depth > 0

        sample_centers = torch.zeros(size=(n_obj, num_sample_contour_point, 2), dtype=torch.int64, device=self.device)
        sample_normals = torch.zeros(size=(n_obj, num_sample_contour_point, 2), dtype=torch.float32, device=self.device)
        for i in range(n_obj):
            total_contour_length_in_pixel, contour, normal = \
                self.generate_valid_contour(mask[i], image_size, k_min_contour_length, k_contour_normal_approx_radius)

            if total_contour_length_in_pixel == 0:
                print('----- total_contour_length_in_pixel == 0 | obj index=%d -----' % (i))
                return False, [], [], [], []

            # random_sample_index = \
            #     (torch.rand(num_sample_contour_point, device=contour.device) * contour.shape[0])\
            #         .long().unsqueeze(-1).expand(-1, 2)
            # sample_center = torch.gather(contour, index=random_sample_index, dim=0)
            # sample_centers[i] = sample_center.type(torch.int64)
            # sample_normal = torch.gather(normal, index=random_sample_index, dim=0)
            # sample_normals[i] = sample_normal
            sample_step = (total_contour_length_in_pixel - 1) // num_sample_contour_point + 1
            contour2 = torch.cat((contour, contour), dim=0)
            sample_center = contour2[::sample_step, :]
            sample_center = sample_center[:num_sample_contour_point, :]
            sample_centers[i] = sample_center.type(torch.int64)
            normal2 = torch.cat((normal, normal), dim=0)
            sample_normal = normal2[::sample_step, :]
            sample_normal = sample_normal[:num_sample_contour_point, :]
            sample_normals[i] = sample_normal

            # Display contour with normal
            # display_image = np.zeros(shape=(image_size[1], image_size[0], 3), dtype=np.uint8)
            # sample_center_np = sample_center.cpu().numpy()
            # sample_normal_np = sample_normal.cpu().numpy()
            # for j in range(num_sample_contour_point):
            #     x = sample_center_np[j][0]
            #     y = sample_center_np[j][1]
            #     cv2.circle(display_image, (x, y), 4, (0, 255, 0), -1)
            #     nx = int(x + sample_normal_np[j][0] * 30)
            #     ny = int(y + sample_normal_np[j][1] * 30)
            #     cv2.line(display_image, (x, y), (nx, ny), (0, 0, 255), 2)
            # cv2.imwrite('./data/contour.png', display_image)
            # import ipdb;
            # ipdb.set_trace();

        sample_centers_index = (sample_centers[..., 1] * image_size[0] + sample_centers[..., 0])
        z = depth.view(n_obj, -1)
        z = torch.gather(z, index=sample_centers_index, dim=1)

        centers_in_view, valid = viewer.camera.image2view(sample_centers, z)
        centers_in_world = viewer.view2world_pose.transform(centers_in_view)
        centers_in_body = body.world2body_pose.transform(centers_in_world)

        normal_z = torch.zeros(size=(n_obj, num_sample_contour_point, 1), device=self.device)
        normals_in_view = torch.cat((sample_normals, normal_z), dim=-1)
        normals_in_world = viewer.view2world_pose.rotate(normals_in_view)
        normals_in_body = body.world2body_pose.rotate(normals_in_world)

        # draw_vertices_to_obj(centers_in_body[0].cpu().numpy(), './data/teapot_in_body.obj')
        # draw_vertices_to_obj(centers_in_body[0].cpu().numpy(), './data/ape_in_body.obj')
        #draw_axis_to_obj_1(viewer.view2world_pose.R.cpu(), viewer.view2world_pose.t.cpu(),
        #                    './data/first_camera_pose.obj')

        pixel_to_meter = centers_in_view[..., 2] / viewer.camera.f[..., 0].unsqueeze(-1)
        foreground_distance, background_distance = \
            self.calculate_line_distances(image_size, mask.cpu().numpy(),
                                          sample_centers.cpu().numpy(), sample_normals.cpu().numpy(),
                                          pixel_to_meter.cpu().numpy())
        # k = 0
        # display_image = np.zeros(shape=(image_size[1], image_size[0], 3), dtype=np.uint8)
        # sample_center_np = sample_centers[k].cpu().numpy()
        # sample_normal_np = sample_normals[k].cpu().numpy()
        # pixel_to_meter_np = pixel_to_meter[k].cpu().numpy()
        # for j in range(num_sample_contour_point):
        #     x = sample_center_np[j][0]
        #     y = sample_center_np[j][1]
        #     cv2.circle(display_image, (x, y), 4, (0, 255, 0), -1)
        #     if j % 20 == 0:
        #         in_len = 30
        #         if foreground_distance[k, j] / pixel_to_meter_np[j] < self.numeric_max:
        #             in_len = foreground_distance[k, j] / pixel_to_meter_np[j]
        #         nx = int(x - sample_normal_np[j][0] * in_len)
        #         ny = int(y - sample_normal_np[j][1] * in_len)
        #         cv2.line(display_image, (x, y), (nx, ny), (0, 0, 255), 2)
        #         out_len = 30
        #         if background_distance[k, j] / pixel_to_meter_np[j] < self.numeric_max:
        #             out_len = background_distance[k, j] / pixel_to_meter_np[j]
        #         nx = int(x + sample_normal_np[j][0] * out_len)
        #         ny = int(y + sample_normal_np[j][1] * out_len)
        #         cv2.line(display_image, (x, y), (nx, ny), (0, 255, 255), 2)
        # cv2.imwrite('./data/contour.png', display_image)
        # import ipdb;
        # ipdb.set_trace();

        return True, centers_in_body.cpu().numpy(), normals_in_body.cpu().numpy(),\
               foreground_distance, background_distance


















