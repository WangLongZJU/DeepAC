from .wrappers import Pose, Camera

class Viewer():
    def __init__(self, image_size, view2world_pose: Pose, camera: Camera,
                 view_coordinate_to_render_coordinate_func=None, device="cpu"):
        assert view2world_pose.num == camera.num
        self.view2world_pose = view2world_pose.to(device)
        self.camera = camera.to(device)
        self.view_coordinate_to_render_coordinate_func = view_coordinate_to_render_coordinate_func
        # the image size must be same in one viewer
        self.image_size = image_size

    @property
    def num(self) -> int:
        return self.view2world_pose.num

    @property
    def world2view_pose(self) -> Pose:
        return self.view2world_pose.inv()

