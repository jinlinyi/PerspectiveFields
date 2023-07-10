# Project Environment
### Core library

PerspectiveFields requires python >= 3.8 and [PyTorch](https://pytorch.org/).


| ***Pro tip:*** *use [mamba](https://github.com/mamba-org/mamba) in place of conda for much faster installs.*
The dependencies can be installed by running:
```bash
git clone git@github.com:jinlinyi/PerspectiveFields.git
# create virtual env
conda create -n perspective python=3.9
conda activate perspective
# install pytorch compatible to your system
conda install pytorch torchvision cudatoolkit -c pytorch
# conda packages
conda install -c conda-forge openexr-python openexr
# pip packages
pip install -r requirements.txt
# install mmcv with mim, I encountered some issue with pip install mmcv :(
mim install mmcv
# install Perspective Fields.
pip install -e .
```

