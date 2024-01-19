from ..dataset.utils import read_template_data
import os
import torch
import numpy as np

if __name__ == "__main__":
    model_dir = "./datasets/MyModel"
    obj_names = ['pikachu']
    obj_template_paths = []
    for obj_name in obj_names:
        template_path = os.path.join(model_dir, obj_name, 'pre_render', obj_name+'.pkl')
        obj_template_paths.append(template_path)
    
    num_sample_contour_points, template_views, orientations_in_body = \
        read_template_data(obj_names, obj_template_paths)

    for obj_name in obj_names:
        num_contour_points = num_sample_contour_points[obj_name]
        template_view = template_views[obj_name]
        orientations = orientations_in_body[obj_name]
        num_template_view = template_view.shape[0] / num_contour_points
        output_data = torch.cat((orientations.flatten(), template_view.flatten())).numpy()
        output_head = np.array([num_contour_points, num_template_view], dtype=np.int32)

        output_data_path = os.path.join(model_dir, obj_name, 'pre_render', obj_name+'.data')
        output_head_path = os.path.join(model_dir, obj_name, 'pre_render', obj_name+'.txt')
        np.savetxt(output_head_path, output_head)
        with open(output_data_path, 'wb') as fp:
            fp.write(output_data)