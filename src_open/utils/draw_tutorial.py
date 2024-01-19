import numpy as np
import torch
import cv2

# vertices: (n, 3), np.array
def draw_vertices_to_obj(vertices, path):
    n = vertices.shape[0]
    lines = ''
    for i in range(n):
        lines += 'v ' + str(vertices[i, 0]) + ' ' + str(vertices[i, 1]) + ' ' + str(vertices[i, 2]) + '\n'

    with open(path, 'w') as f:
        f.write(lines)

def draw_axis(center, direction, sample_num, length, color, output_lines):
    for k in range(sample_num):
        v = center + direction * (float(k) / sample_num) * length
        output_lines += 'v ' + str(v[0].item()) + ' ' + str(v[1].item()) + ' ' + str(v[2].item()) + \
                        ' ' + str(color[0].item()) + ' ' + str(color[1].item()) + ' ' + str(color[2].item()) + '\n'

    return output_lines

# camera_pose: (n, 4, 4), tensor
def draw_axis_to_obj(camera_pose: torch.Tensor, path):
    lines = ''
    length = 100
    sample_num = 30
    color = torch.tensor([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    n = camera_pose.shape[0]
    for i in range(n):
        center = camera_pose[i, :3, 3]
        for j in range(3):
            direction = camera_pose[i, :3, j]
            lines = draw_axis(center, direction, sample_num, length, color[j], lines)

    with open(path, 'w') as f:
        f.write(lines)

# camera_pose_R: (n, 3, 3), tensor
# camera_pose_t: (n, 3), tensor
def draw_axis_to_obj_1(camera_pose_R, camera_pose_t, path):
    lines = ''
    length = 0.1
    sample_num = 30
    color = torch.tensor([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    n = camera_pose_R.shape[0]
    for i in range(n):
        center = camera_pose_t[i]
        for j in range(3):
            direction = camera_pose_R[i, :3, j]
            lines = draw_axis(center, direction, sample_num, length, color[j], lines)

    with open(path, 'w') as f:
        f.write(lines)

# centers: (n, 2), np.array
# valid: (n), np.array
# normals: (n, 2), np.array
# foredist: (n), np.array
# backdist: (n), np.array
def draw_correspondence_lines_in_image_with_weight(image, centers, valid, weight, normals=None,
                                                   foredist=None, backdist=None, center_color=(255, 0, 0)):
    n = centers.shape[0]
    for i in range(0, n, 1):
        color = (int(center_color[0]*weight[i]), int(center_color[1]*weight[i]), int(center_color[2]*weight[i]))
        if valid[i]:
            center = centers[i]
            normal = normals[i]
            cv2.circle(image, (int(center[0]), int(center[1])), 1, color, -1)
            if foredist is not None:
                if isinstance(foredist, np.int64) or isinstance(foredist, int):
                    normal_len = foredist
                else:
                    normal_len = foredist[i]
            else:
                normal_len = 10
            ns = center - normal * normal_len
            ne = center + normal * normal_len
            cv2.line(image, (int(ns[0]), int(ns[1])), (int(ne[0]), int(ne[1])), (0, int(255*weight[i]), 0), 1)

    return image

# centers: (n, 2), np.array
# valid: (n), np.array
# normals: (n, 2), np.array
# foredist: (n), np.array
# backdist: (n), np.array
def draw_correspondence_lines_in_image(image, centers, valid, normals=None,
                                       foredist=None, backdist=None, center_color=(255, 0, 0)):
    n = centers.shape[0]
    for i in range(0, n, 1):
        if valid[i]:
            center = centers[i]
            normal = normals[i]
            cv2.circle(image, (int(center[0]), int(center[1])), 1, center_color, -1)
            if foredist is not None:
                if isinstance(foredist, np.int64) or isinstance(foredist, int):
                    normal_len = foredist
                else:
                    normal_len = foredist[i]
            else:
                normal_len = 10
            ns = center - normal * normal_len
            ne = center + normal * normal_len
            cv2.line(image, (int(ns[0]), int(ns[1])), (int(ne[0]), int(ne[1])), (0, 255, 0), 1)

    return image

def draw_centers_in_image(image, centers, valid, center_color=(255, 0, 0), confidence=None):
    n = centers.shape[0]
    for i in range(0, n, 1):
        if valid[i]:
            center = centers[i]
            if confidence is None:
                color = center_color
            else:
                color = (int(center_color[0] * confidence[i]),
                         int(center_color[1] * confidence[i]),
                         int(center_color[2] * confidence[i]))

            cv2.circle(image, (int(center[0]), int(center[1])), 1, color, -1)

    return image


if __name__ == "__main__":
    pose = torch.tensor([0.9990760087966919, -0.04063175991177559, 0.014005012810230255, 0.033269185572862625,
                        -0.04240379109978676, -0.8788546323776245, 0.4752015173435211, 0.03613205999135971,
                        -0.006999903358519077, -0.4753563106060028, -0.879765510559082, 0.42635953426361084])
    pose = pose.view(3, 4).unsqueeze(0)
    draw_axis_to_obj(pose, './axis.obj')