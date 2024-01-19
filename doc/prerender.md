# Pre-generate Coutour Points for Your Custom Data
1.Update the config in `REPO/src_open/configs/prerender/MyModel.yaml`:
```yaml
type: prerender

prerender_method: MyModel
data_dir: "data/demo"      # where your mesh.obj saves
batch_size: 1
body_names: ['Cat']        # your mesh.obj name
sphere_radius: 0.8
maximum_body_diameter: 0.3
geometry_unit_in_meter: 0.001  # the mesh.obj will be transformed in the unit in meter
                               # for example: if the mesh.obj is in cm, then geometry_unit_in_meter = 0.01;
                               # if the mesh.obj is in mm, then geometry_unit_in_meter = 0.001;
    
image_size: 2000
image_border_size: 20
n_divide: 4
normalize_to_origin: False # Fixed
num_sample_contour_point: 200
k_min_contour_length: 15
k_contour_normal_approx_radius: 3
output_mask: False         # output the mask of each viewpoints if True
output_depth: False        # Fixed
output_depth_vertex: False # output the inverse projected vertex of the depth of each viewpoints if True
```
2.Use the command as following:
```python
python -m src_open.run +prerender=MyModel
```
The pre-generated contour points and other optional items will be saved in the same directory of the mesh, like:
```shell
|--- demo
|     |--- Cat.obj
|     |--- Cat
|     |     |--- pre_render
|     |     |      |--- Cat.pkl
|     |     |      |--- mask
|     |     |      |--- depth_vertex
```