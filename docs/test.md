
# Datasets

In our paper, we tested the **PersNet-360Cities** model on images from publicly available datasets [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html) and [TartanAir](http://theairlab.org/tartanair-dataset/). Results can be found in Table 1.

To download Stanford2d3d dataset:
First agree to their data sharing and usage term: [link](https://docs.google.com/forms/d/e/1FAIpQLScFR0U8WEUtb7tgjOhhnl31OrkEs73-Y8bQwPeXgebqVKNMpQ/viewform?c=0&w=1). 
```bash
https://www.dropbox.com/sh/ycd4hv0t1nqagub/AACjqZ2emGw7L-aAJ1rmpX4-a
```

Download TartanAir dataset:
```bash
https://www.dropbox.com/sh/7tev8uqnnjfhzhb/AAD9y_d1DCcoZ-AQDEQ1tn0Ua
```
Extract the datasets under `perspectiveField/datasets`.

We also tested the **PersNet_paramnet-GSV-centered** and **PersNet_Paramnet-GSV-uncentered** models on centered and uncentered images from Google Street View (GSV). Results can be found in Tables 3 and 4.

Download GSV datasets:
```bash
wget https://www.dropbox.com/s/plcmcza8vfmmpkm/google_street_view_191210.tar
wget https://www.dropbox.com/s/9se3lrpljd59cod/gsv_test_crop_uniform.tar
```
Extract the datasets under `perspectiveField/datasets`.


# Testing PerspectiveNet + ParamNet

## PerspectiveNet:

First, to test PerspectiveNet, provide a dataset name corresponding to a name/path pair from `perspective2d/data/datasets/builtin.py`. Create and provide an output folder under `perspectiveField/exps`. Choose a model and provide the path to the config file and weights, both of which should be under `perspectiveField/models`.

Example:
```bash
python -W ignore demo/test_warp.py \
--dataset stanford2d3d_test \
--output ./exps/persnet_360_cities_test \
--config-file ./models/cvpr2023.yaml \
--opts MODEL.WEIGHTS ./models/cvpr2023.pth
```

## ParamNet:

To test ParamNet, again provide a dataset name, output folder, and a path to config and model weights, just as with PerspectiveNet.

Example:
```bash
python -W ignore demo/test_param_network.py \
--dataset gsv_test  \
--output ./exps/paramnet_gsv_test \
--config-file ./models/paramnet_gsv_rpf.yaml \
--opts MODEL.WEIGHTS ./models/paramnet_gsv_rpf.pth
```
