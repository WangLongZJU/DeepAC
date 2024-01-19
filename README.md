# Deep Active Contours for Real-time 6-DoF Object Tracking
### [Project Page](https://zju3dv.github.io/deep_ac/) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Deep_Active_Contours_for_Real-time_6-DoF_Object_Tracking_ICCV_2023_paper.pdf)
<br/>

> Deep Active Contours for Real-time 6-DoF Object Tracking                                                                             
> [Long Wang](https://wanglongzju.github.io/wanglong.github.io/)<sup>\*</sup>, [Shen Yan]()<sup>\*</sup>, [Jianan Zhen](), [Yu Liu](), [Maojun Zhang](), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Xiaowei Zhou](https://xzhou.me)                              
> ICCV 2023


<!-- ![demo_vid](assets/demo.gif) # TODO -->

## Installation
```shell
conda create -n deepac python=3.8
conda activate deepac
# install torch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1"
# install other dependencies
pip install -r requirements.txt
```
Download the [pretrained models](https://drive.google.com/file/d/1B7qolNwPMhLlUEcN5Bi8iDc2XxrR-M4e/view?usp=sharing), and place them in `workspace/train_bop_deepac`, like:
```shell
|--- workspace/train_bop_deepac
|       |--- logs-2024-01-08-15-52-47
|       |         |--- train_cfg.yml
|       |         |--- model_last.ckpt
```

## Demo
After the installation, you can refer to [this page](doc/demo.md) to run the demo with your custom data.

## Training and Evaluation
### Dataset setup
1. DeepAC uses eight datasets from the [BOP challenge](https://bop.felk.cvut.cz/challenges/), namely HOPE, IC-BIN, IC-MI, T-LESS, TUD-L, LM, YCB-V, and RU-APC for training and validation. You can download the datasets from [here](https://bop.felk.cvut.cz/datasets/). For evaluation, DeepAC uses [RBOT](https://www.mi.hs-rm.de/~schwan/research/RBOT/), [BCOT](https://ar3dv.github.io/BCOT-Benchmark/) and [OPT](http://media.ee.ntu.edu.tw/research/OPT/) datasets.
2. DeepAC uses the data [SUN2012](https://drive.google.com/file/d/1tEYDbit4imuosqrbeI32H4cgwuLEQLaI/view?usp=drive_link) to change the background for augmentation.
3. You can extract these datasets into `$/your/path/to/datasets` and the directory should be organized in the following structure:
    ```shell
    |--- /your/path/to/datasets
    |       |--- BOP
    |       |      |--- hope
    |       |      |--- icbin
    |       |      |--- icmi
    |       |      |--- tless
    |       |      |--- icmi
    |       |      |--- lm
    |       |      |--- ycbv
    |       |      |--- ruapc
    |       |--- RBOT
    |       |--- BCOT
    |       |--- OPT
    |       |--- SUN2012
    ```

4. Build the dataset symlinks
    ```shell
    REPO=/path/to/DeepAC
    ln -s /your/path/to/datasets $REPO/datasets
    ```

### Data Preprocess
DeepAC need to pre-generate contour points from various perspectives of the object, which can avoid mesh rendering duiring the tracking. You can use the following command:
```shell
sh ./script/prerender_for_training.sh
sh ./script/prerender_for_evaluation.sh
```
**We strongly suggest utilizing the mesh.obj which is simplified in MeshLab. For example the obj in OPT dataset has million vertices and faces, but generating contour points does not need these numerous items**
### Training
```shell
python -m src_open.run +train=train_BOP_deepac
```
The results of training will be automatically saved in the `REPO/workspace/train_BOP_deepac` like:
```shell
|--- workspace/train_bop_deepac
|       |--- logs-2024-01-08-15-52-47
|       |         |--- train_cfg.yml
|       |         |--- model_0.ckpt
|       |         |--- model_last.ckpt
|       |--- logs-2024-01-08-19-31-15
```
### Evaluation
1. For RBOT, update the `load_cfg` and `load_model` of the `REPO/src_open/configs/test_deepac/test_RBOT.yaml` to load the trained model. Use command as following:
    ```python
    python -m src_open.run +test_deepac=test_RBOT
    ```
    The results of evaluation will be automatically saved in the `REPO/workspace/test_rbot_deepac`, like:
    ```shell
    |--- workspace/test_rbot_deepac
    |       |--- logs-2024-01-08-19-09-53
    |       |         |--- ape_a_regular_pose.txt # the poses of this sequence
    |       |         |--- ape_a_regular.avi      # the visual result of this seqence
    |       |         |--- test_results.json      # the total results
                                                  # you can find the "total" in this .json about whole cm-degree and ADD metrics
    |       |--- logs-2024-01-08-23-04-27
    ```
    If you want to get the results for each RBOT scene, such as "Regular", "Dynamic Light", "Noisy" and "Unmodeled Occlusion", you need to update the `load_json` in the `REPO/src_open/configs/test_json/test_json.yaml` and use the following command:
    ```shell
    python -m src_open.run +test_json=test_json
    ```
    The results will be printed to the terminal.
2. For BCOT, update the `load_cfg` and `load_model` of the `REPO/src_open/configs/test_deepac/test_BCOT.yaml` to load the trained model. Use command as following:
    ```python
    python -m src_open.run +test_deepac=test_BCOT
    ```
    The results of evaluation will be automatically saved in the `REPO/workspace/test_bcot_deepac`, like:
    ```shell
    |--- workspace/test_bcot_deepac
    |       |--- logs-2024-01-16-14-24-17
    |       |         |--- 3D Touch_complex_movable_handheld_pose.txt      # the poses of this sequence
    |       |         |--- 3D Touch_complex_movable_handheld_pose.avi      # the visual result of this seqence
    |       |         |--- test_results.json      # the total results
                                                  # you can find the "total" in this .json about whole cm-degree and ADD metrics
    |       |--- logs-2024-01-08-23-04-27
    ```
3. For OPT, use command as following for each object:
    ```python
    python -m src_open.run +test_deepac=test_OPT 'data.test_obj_names=[bike]'
    python -m src_open.run +test_deepac=test_OPT 'data.test_obj_names=[chest]'
    python -m src_open.run +test_deepac=test_OPT 'data.test_obj_names=[house]'
    python -m src_open.run +test_deepac=test_OPT 'data.test_obj_names=[ironman]'
    python -m src_open.run +test_deepac=test_OPT 'data.test_obj_names=[jet]'
    python -m src_open.run +test_deepac=test_OPT 'data.test_obj_names=[soda]'
    ```
    The results of evaluation will be automatically saved in the `REPO/workspace/test_opt_deepac`, like: 
    ```shell
    |--- workspace/test_bcot_deepac
    |       |--- logs-2024-01-16-14-24-17
    |       |         |--- bike_fl_b_pose.txt      # the poses of this sequence
    |       |         |--- bike_fl_b_pose.avi      # the visual result of this seqence
    |       |         |--- test_results.json      # the total results
                                                  # you can find the "total" in this .json about whole cm-degree and ADD metrics
    |       |         |--- logs.txt               # you can find AUC score in this txt
    |       |--- logs-2024-01-08-23-04-27
    ```

## Deployment
We provide the deployment code to convert pytorch models to mlmodel via [coremltools](https://github.com/apple/coremltools). You just need to update the the `load_cfg` and `load_model` of the `REPO/src_open/configs/deploy_deepac`, and then use the following command to deploy the model:
```shell
python -m src_open.run +deploy_deepac=deploy
```
The converted models will be automatically saved in the `REPO/workspace/deploy_deepac`, like:
```shell
|--- workspace/deploy_deepac
|       |--- logs-2024-01-10-15-11-06
|       |         |--- boundary_predictor.mlpackage 
|       |         |--- contour_feature_extractor0.mlpackage      
|       |         |--- contour_feature_extractor1.mlpackage      
|       |         |--- contour_feature_extractor2.mlpackage
|       |         |--- derivative_calculator.mlpackage
|       |         |--- extractor.mlpackage
|       |         |--- histogram.mlpackage
```
You can use these models to develop your application in [Xcode](https://developer.apple.com/xcode/). You can refer to [Core ML](https://developer.apple.com/documentation/coreml) for more information.

## Citation
If you find this code useful for your research, please use the following BibTeX entry.
```bibtex
@inproceedings{wang2023deep,
  title={Deep Active Contours for Real-time 6-DoF Object Tracking},
  author={Long Wang and Shen Yan and Jianan Zhen and Yu Liu and Maojun Zhang and Guofeng Zhang and Xiaowei Zhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## Acknowledgement
Thanks for these great optimization-based methods [RBOT](https://github.com/henningtjaden/RBOT), [3D Object Tracking](https://github.com/DLR-RM/3DObjectTracking/tree/master) and [LDT3D](https://github.com/cvbubbles/nonlocal-3dtracking), which inspired us to develop this method.