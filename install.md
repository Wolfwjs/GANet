## Installation

### Requirements

- Linux
- Python 3.7
- PyTorch 1.6
- CUDA 10.1 (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv)


### Install mmdetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n openmm python=3.7 -y
conda activate openmm
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
(目前用的是以上版本)
```

c. Clone repository
```shell
git clone [待上传到gitlab]
cd [project]
```

d. Install build requirements and then install mmdetection.
(We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

```shell
pip install -r requirements/build.txt
python setup.py develop
(需要srun到集群上setup)
```

