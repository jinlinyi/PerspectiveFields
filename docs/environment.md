# Project Environment
```bash
# Python 3.9.7
conda create -n perspective python=3.9
conda activate perspective
# pytorch 1.10.0 with CUDA 11.3  (see torch.version.cuda)
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge openexr-python openexr
# Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
# others
pip install gitpython opencv-contrib-python albumentations pyequilib==0.3.0 skylibs timm mmcv h5py tensorboard setuptools==59.5.0
# install local packages
cd perspectiveField
pip install -e .
cd ..
```
