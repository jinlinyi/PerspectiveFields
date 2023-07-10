# Models

NOTE: Extract any model weights under `./models`
- **PersNet-360Cities**: This model was trained on perspective images cropped from indoor, natural, and street view panoramas from [360cities](https://www.360cities.net) and is trained to predict Perspective Fields.
    - Download model weights:
    ```bash
    wget https://www.dropbox.com/s/czqrepqe7x70b7y/cvpr2023.pth
    ```
    The config file can be found here: `./models/cvpr2023.yaml`.

- **PersNet_paramnet-GSV-centered**: This model was trained on the Google Street View dataset and is trained to predict Perspective Fields + camera parameters (roll, pitch, vfov), assuming a centered principal point.
    - Download model weights:
    ```bash
    wget https://www.dropbox.com/s/g6xwbgnkggapyeu/paramnet_gsv_rpf.pth
    ```
    The config file can be found here: `./models/paramnet_gsv_rpf.yaml`

- **PersNet_Paramnet-GSV-uncentered**: This model was trained on the Google Street View dataset and is trained to predict Perspective Fields + camera parameters (roll, pitch, vfov, cx, cy). No assumption is made on the location of the principle point.
    - Download model weights:
    ```bash
    wget https://www.dropbox.com/s/ufdadxigewakzlz/paramnet_gsv_rpfpp.pth
    ```
    The config file can be found here: `./models/paramnet_gsv_rpfpp.yaml`

- **NEW:Paramnet-360Cities-edina-centered**: This model was trained on data from both [360cities](https://www.360cities.net) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md). The training data consists of a diverse set of indoor, outdoor, natural, and egocentric data. The model is trained to predict Perspective Fields + camera parameters (roll, pitch, vfov), assuming a centered principal point.
    - Download model weights:
    ```bash
    wget https://www.dropbox.com/s/z2dja70bgy007su/paramnet_360cities_edina_rpf.pth
    ```
    The config file can be found here: `./models/paramnet_360cities_edina_rpf.yaml`

- **NEW:Paramnet-360Cities-edina-uncentered**: This model was trained on data from both [360cities](https://www.360cities.net) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md). The training data consists of a diverse set of indoor, outdoor, natural, and egocentric data. The model is trained to predict Perspective Fields + camera parameters (roll, pitch, vfov, cx, cy). No assumption is made on the location of the principle point.
    - Download model weights:
    ```bash
    wget https://www.dropbox.com/s/nt29e1pi83mm1va/paramnet_360cities_edina_rpfpp.pth
    ```
    The config file can be found here: `./models/paramnet_360cities_edina_rpfpp.yaml`


# Datasets

In our paper, we tested the **PersNet-360Cities** model on images from publicly available datasets [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html) and [TartanAir](http://theairlab.org/tartanair-dataset/). Results can be found in Table 1.

Download Stanford2d3d dataset:
```bash
wget https://www.dropbox.com/sh/ycd4hv0t1nqagub/AACjqZ2emGw7L-aAJ1rmpX4-a?dl=0
```

Download TartanAir dataset:
```bash
wget https://www.dropbox.com/sh/7tev8uqnnjfhzhb/AAD9y_d1DCcoZ-AQDEQ1tn0Ua?dl=0
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

First, to test PerspectiveNet, provide a dataset name corresponding to a name/path pair from `perspective2d/data/datasets/builtin.py`. Create and provide an output folder under `./exps`. Choose a model and provide the path to the config file and weights, both of which should be under `./models`.

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
