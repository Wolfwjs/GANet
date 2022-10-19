# GANet
This repo is the PyTorch implementation for our paper:

[A Keypoint-based Global Association Network for Lane Detection](https://arxiv.org/abs/2204.07335). Accepted by CVPR 2022.

![img.png](images/ganet.png)
In this paper, we propose a Global Association Network (GANet) to formulate the lane detection problem from a new perspective, where each keypoint is directly regressed to the starting point of the lane line instead of point-by-point extension. Concretely, the association of keypoints to their belonged lane line is conducted by predicting their offsets to the corresponding starting points of lanes globally without dependence on each other, which could be done in parallel to greatly improve efficiency. In addition, we further propose a Lane-aware Feature Aggregator (LFA), which adaptively captures the local correlations between adjacent keypoints to supplement local information to the global association.

## Installation
 1. Create a conda virtual environment and activate it.
    ```shell
    conda create -n ganet python=3.8 -y
    conda activate ganet
    conda install pytorch=1.8 torchvision=0.9.0 cudatoolkit=10.1 -c pytorch
    pip install openmim
    mim install mmcv-full
    mim install mmdet
    pip install -r requirements.txt
    ```
 2. Clone this repository and enter it:
    ```Shell
    git clone https://github.com/Wolfwjs/GANet.git
    cd GANet
    python setup.py develop
    ```

## Dataset
[Prepare Dataset](dataset.md)

## Evaluation
Here are our GANet models (released on April 24th, 2022):

### CULane
| Version |   Backbone    | FPS |  F1   | Weights                                                                                                          | 
|:-------:|:-------------:|:---:|:-----:|------------------------------------------------------------------------------------------------------------------|
|  Small  |   ResNet18    | 153 | 78.79 | [ganet_culane_resnet18.pth](https://drive.google.com/file/d/1-L7cfKYeiQVxaDlN9dxnNH9cWp5wIt7f/view?usp=sharing)  | 
| Medium  |   ResNet-34   | 127 | 79.39 | [ganet_culane_resnet34.pth](https://drive.google.com/file/d/1fJQPecJn1FVXAux2YTIEPQHhlNv7sHC9/view?usp=sharing)  | 
|  Large  |  ResNet-101   | 63  | 79.63 | [ganet_culane_resnet101.pth](https://drive.google.com/file/d/1X49SLAbzrFTjiRzp_YUiP7eOmFCToIJM/view?usp=sharing) | 

### TuSimple
| Version |   Backbone    | FPS |  F1   | Weights                                                                                                            | 
|:-------:|:-------------:|:---:|:-----:|--------------------------------------------------------------------------------------------------------------------|
|  Small  |   ResNet18    | 153 | 97.71 | [ganet_tusimple_resnet18.pth](https://drive.google.com/file/d/1Zbo0CdjksWK46gpuuB6NMvPxc0Zu50fD/view?usp=sharing)  | 
| Medium  |   ResNet-34   | 127 | 97.68 | [ganet_tusimple_resnet34.pth](https://drive.google.com/file/d/1NpnWQQJPrmKHe9LAQkej3RKi9qq1allC/view?usp=sharing)  | 
|  Large  |  ResNet-101   | 33  | 97.45 | [ganet_tusimple_resnet101.pth](https://drive.google.com/file/d/1b5kPp79MabCRH06CEGXvj_XW11SR8ROM/view?usp=sharing) | 

To evalute the model, download the corresponding weights file into the `[CHECKPOINT]` directory and run the following commands.

```shell
# For example, model = ganet-small 
CUDA_VISIBLE_DEVICES=1,2,3 GPUS=3 bash tools/dist_test.sh projects/cfgs/tusimple/final_exp_res18_s8.py [CHECKPOINT] --eval
```
We use the official evaluation tools of [CULane](https://github.com/XingangPan/SCNN) and [TuSimple](https://github.com/TuSimple/tusimple-benchmark/tree/master/evaluate) to evaluate the results. And we include them in `tools` directory which may be helpful for you.
## Training
```shell
# For example, model = ganet-small 
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/dist_train.sh projects/cfgs/culane/final_exp_res18_s8.py
```
# Citation
If you find this repo useful for your research, please cite
```
@inproceedings{ganet-cvpr2022,
  author    = {Jinsheng Wang, Yinchao Ma, Shaofei Huang, Tianrui Hui, Fei Wang, Chen Qian, Tianzhu Zhang},
  title     = {A Keypoint-based Global Association Network for Lane Detection},
  booktitle = {CVPR},
  year      = {2022},
}
```

# Contact

For questions about our paper or code, please contact [Jinsheng Wang](mailto:jswang@stu.pku.edu.cn) or [Yinchao Ma](mailto:imyc@mail.ustc.edu.cn)
