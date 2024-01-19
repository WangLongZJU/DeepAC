import torch

from .wrappers import Pose
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
)

class Body():

    def __init__(self, name, obj_path, geometry_unit_in_meter, maximum_body_diameter,
                 body_coordinate_to_render_coordinate_func=None, normalize_to_origin=True, device="cpu"):

        if not isinstance(obj_path, list):
            obj_path = list([obj_path])

        self.name = name
        self.device = device
        self.n_body = len(obj_path)
        self.obj_path = obj_path
        self.body2world_pose = Pose.from_identity(len(obj_path)).to(device)
        self.body_coordinate_to_render_coordinate_func = body_coordinate_to_render_coordinate_func

        verts = []
        faces = []
        for path in obj_path:
            assert '.ply' in path or '.obj' in path
            if '.obj' in path:
                vert, faces_idx, _ = load_obj(path)
                face = faces_idx.verts_idx
            if '.ply' in path:
                vert, face = load_ply(path)
            vert *= geometry_unit_in_meter
            verts.append(vert.to(device))
            faces.append(face.to(device))
        self.meshes = Meshes(
            verts=verts,
            faces=faces,
        )

        # verts, faces_idx, _ = load_obj(obj_path[0])
        # verts *= geometry_unit_in_meter
        # faces = faces_idx.verts_idx
        # verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        # textures = TexturesVertex(verts_features=verts_rgb.to(device))
        # self.meshes = Meshes(
        #     verts=[verts.to(device)],
        #     faces=[faces.to(device)],
        #     textures=textures,
        # )

        # self.meshes = load_objs_as_meshes(obj_path, device=device, load_textures=False)
        self.meshes_bounding_boxes = self.meshes.get_bounding_boxes()
        self.body_diameter = self.meshes_bounding_boxes[..., 1] - self.meshes_bounding_boxes[..., 0]
        # self.maximum_body_diameter, _ = torch.max(self.body_diameter, dim=1)
        self.maximum_body_diameter = maximum_body_diameter

        if normalize_to_origin:
            bounding_box_centers = (self.meshes_bounding_boxes[..., 1] + self.meshes_bounding_boxes[..., 0]) / 2
            self.body2world_pose._data[..., -3:] = -bounding_box_centers

    @property
    def world2body_pose(self) -> Pose:
        return self.body2world_pose.inv()