# Run Demo on Custom Data
## Inference
We provide a demo for inference and the data is saved in `REPO/data/demo`:
```shell
|--- demo
|     |--- img
|     |     |--- 0000.png
|     |     |--- 0001.png
|     |     |--- ...
|     |     |--- pose.txt
|     |--- Cat
|     |     |--- pre_render
|     |     |        |--- Cat.pkl
|     |--- Cat.obj
|     |--- K.txt
```
The `Cat.pkl` is the pre-generated contour points of the `Cat.obj` mesh. You can follow the [doc](prerender.md) to generate this `.pkl` for your mesh. Use the following command to run the demo:
```python
python -m src_open.run +demo=demo
```
The result will be saved in `REPO/workspace/demo`

## Inference for your custom data
### Step 1: Organize the file structure of your data
```shell
|--- your_data_dir
|     |--- img
|     |     |--- 0000.png
|     |     |--- 0001.png
|     |     |--- ...
|     |     |--- pose.txt
|     |--- your_mesh
|     |     |--- pre_render
|     |     |      |--- your_mesh.pkl
|     |--- your_mesh.obj
|     |--- your_K.txt
```
### Step 2: Update the config in `REPO/src_open/configs/demo/demo.yaml`:
```yaml
type: demo

save_dir: ${work_dir}/workspace/demo
load_cfg: ${work_dir}/workspace/train_bop_deepac/logs-2024-01-08-15-52-47/train_cfg.yml
load_model: ${work_dir}/workspace/train_bop_deepac/logs-2024-01-08-15-52-47/model_last.ckpt
# -----------------

data_dir: data/demo              # update to your data path
obj_name: Cat                    # update to your mesh name
geometry_unit_in_meter: 0.001    # update to your geometry unit
output_video: true
output_size: [320, 320]

fore_learn_rate: 0.2
back_learn_rate: 0.2
gpu_id: '0' # 0, 1
```
### Step 2: Use the following command:
```python
python -m src_open.run +demo=demo
```
The result will be saved in `REPO/workspace/demo`