<!-- omit in toc -->
Perspective Fields for Single Image Camera Calibration
================================================================
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/jinlinyi/PerspectiveFields)

###  [Project Page](https://jinlinyi.github.io/PerspectiveFields/)  | [Paper](https://arxiv.org/abs/2212.03239) | [Live Demo 🤗](https://huggingface.co/spaces/jinlinyi/PerspectiveFields)

CVPR 2023 (✨Highlight)
<h4>

[Linyi Jin](https://jinlinyi.github.io/)<sup>1</sup>, [Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>, [Yannick Hold-Geoffroy](https://yannickhold.com/)<sup>2</sup>, [Oliver Wang](http://www.oliverwang.info/)<sup>2</sup>, [Kevin Matzen](http://kmatzen.com/)<sup>2</sup>, [Matthew Sticha](https://www.linkedin.com/in/matthew-sticha-746325202/)<sup>1</sup>, [David Fouhey](https://web.eecs.umich.edu/~fouhey/)<sup>1</sup>

<span style="font-size: 14pt; color: #555555">
 <sup>1</sup>University of Michigan, <sup>2</sup>Adobe Research
</span>
</h4>
<hr>

<p align="center">

![alt text](assets/teaser-field.jpg)
</p>
We propose Perspective Fields as a representation that models the local perspective properties of an image. Perspective Fields contain per-pixel information about the camera view, parameterized as an up vector and a latitude value. 

<p align="center">
<img height="100" alt="swiping-1" src="assets/swiping-1.gif"> <img height="100" alt="swiping-2" src="assets/swiping-2.gif"> <img height="100" alt="swiping-3" src="assets/swiping-3.gif"> <img height="100" alt="swiping-4" src="assets/swiping-4.gif">
</p>

📷 From Perspective Fields, you can also get camera parameters if you assume certain camera models. We provide models to recover camera roll, pitch, fov and principal point location.

<p align="center">
  <img src="assets/vancouver/IMG_2481.jpg" alt="Image 1" height="200px" style="margin-right:10px;">
  <img src="assets/vancouver/pred_pers.png" alt="Image 2" height="200px" style="margin-center:10px;">
  <img src="assets/vancouver/pred_param.png" alt="Image 2" height="200px" style="margin-left:10px;">
</p>

<!-- omit in toc -->
Updates
------------------
- [April 2024]: 🚀 We've launched an inference version (`main` branch) with minimal dependencies. For training and evaluation, please checkout [`train_eval` branch](https://github.com/jinlinyi/PerspectiveFields/tree/train_eval). 
- [July 2023]: We released a new model trained on [360cities](https://www.360cities.net/) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md) dataset, consisting of indoor🏠, outdoor🏙️, natural🌳, and egocentric👋 data!
- [May 2023]: Live demo released 🤗. https://huggingface.co/spaces/jinlinyi/PerspectiveFields. Thanks Huggingface for funding this demo!

<!-- omit in toc -->
Table of Contents
------------------
- [Environment Setup](#environment-setup)
  - [Inference](#inference)
  - [Train / Eval](#train--eval)
- [Demo](#demo)
- [Model Zoo](#model-zoo)
- [Coordinate Frame](#coordinate-frame)
- [Camera Parameters to Perspective Fields](#camera-parameters-to-perspective-fields)
- [Visualize Perspective Fields](#visualize-perspective-fields)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)


[1]: ./docs/environment.md
[2]: ./jupyter-notebooks/camera2perspective.ipynb
[3]: ./jupyter-notebooks/predict_perspective_fields.ipynb
[4]: ./jupyter-notebooks/perspective_paramnet.ipynb
[5]: ./docs/train.md
[6]: ./docs/test.md
[7]: ./docs/models.md



## Environment Setup
### Inference
PerspectiveFields requires python >= 3.8 and [PyTorch](https://pytorch.org/).
| ***Pro tip:*** *use [mamba](https://github.com/conda-forge/miniforge) in place of conda for much faster installs.*
```bash
# install pytorch compatible to your system https://pytorch.org/get-started/previous-versions/
conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/jinlinyi/PerspectiveFields.git
```
Alternatively, install the package locally,
```bash
git clone git@github.com:jinlinyi/PerspectiveFields.git
# create virtual env
conda create -n perspective python=3.9
conda activate perspective
# install pytorch compatible to your system https://pytorch.org/get-started/previous-versions/
# conda install pytorch torchvision cudatoolkit -c pytorch
conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
# install Perspective Fields.
cd PerspectiveFields
pip install -e .
```

### Train / Eval
For training and evaluation, please checkout the [`train_eval` branch](https://github.com/jinlinyi/PerspectiveFields/tree/train_eval).


## Demo
Here is a minimal script to run on a single image, see [`demo/demo.py`](demo/demo.py):
```python
import cv2
from perspective2d import PerspectiveFields
# specify model version
version = 'Paramnet-360Cities-edina-centered'
# load model
pf_model = PerspectiveFields(version).eval().cuda()
# load image
img_bgr = cv2.imread('assets/imgs/cityscape.jpg')
# inference
predictions = pf_model.inference(img_bgr=img_bgr)
```
- Or checkout [Live Demo 🤗](https://huggingface.co/spaces/jinlinyi/PerspectiveFields). 
- Notebook to [Predict Perspective Fields](./notebooks/predict_perspective_fields.ipynb). 


## Model Zoo
| Model Name and Weights                                                                                                    | Training Dataset                                                                                                          | Config File                                  | Outputs                                                           | Expected input                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [NEW][Paramnet-360Cities-edina-centered](https://www.dropbox.com/s/z2dja70bgy007su/paramnet_360cities_edina_rpf.pth)       | [360cities](https://www.360cities.net/) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md) | [paramnet_360cities_edina_rpf.yaml](models/paramnet_360cities_edina_rpf.yaml) | Perspective Field + camera parameters (roll, pitch, vfov)         | Uncropped, indoor🏠, outdoor🏙️, natural🌳, and egocentric👋 data                              |
| [NEW][Paramnet-360Cities-edina-uncentered](https://www.dropbox.com/s/nt29e1pi83mm1va/paramnet_360cities_edina_rpfpp.pth)  | [360cities](https://www.360cities.net/) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md) | [paramnet_360cities_edina_rpfpp.yaml](models/paramnet_360cities_edina_rpfpp.yaml) | Perspective Field + camera parameters (roll, pitch, vfov, cx, cy) | Cropped, indoor🏠, outdoor🏙️, natural🌳, and egocentric👋 data                                |
| [PersNet-360Cities](https://www.dropbox.com/s/czqrepqe7x70b7y/cvpr2023.pth)                                               | [360cities](https://www.360cities.net)                                                                                    | [cvpr2023.yaml](models/cvpr2023.yaml)              | Perspective Field                                                 | Indoor🏠, outdoor🏙️, and natural🌳 data.                                                     |
| [PersNet_paramnet-GSV-centered](https://www.dropbox.com/s/g6xwbgnkggapyeu/paramnet_gsv_rpf.pth)                           | [GSV](https://research.google/pubs/pub36899/)                                                                             | [paramnet_gsv_rpf.yaml](models/paramnet_gsv_rpf.yaml)      | Perspective Field + camera parameters (roll, pitch, vfov)         | Uncropped, street view🏙️ data.                                                              |
| [PersNet_Paramnet-GSV-uncentered](https://www.dropbox.com/s/ufdadxigewakzlz/paramnet_gsv_rpfpp.pth)                       | [GSV](https://research.google/pubs/pub36899/)                                                                             | [paramnet_gsv_rpfpp.yaml](models/paramnet_gsv_rpfpp.yaml)    | Perspective Field + camera parameters (roll, pitch, vfov, cx, cy) | Cropped, street view🏙️ data.                                                               |

## Coordinate Frame

<p align="center">

![alt text](assets/coordinate.png)

`yaw / azimuth`: camera rotation about the y-axis
`pitch / elevation`: camera rotation about the x-axis
`roll`: camera rotation about the z-axis

Extrinsics: `rotz(roll).dot(rotx(elevation)).dot(roty(azimuth))`

</p>


## Camera Parameters to Perspective Fields
Checkout [Jupyter Notebook](./notebooks/camera2perspective.ipynb). 
Perspective Fields can be calculated from camera parameters. If you prefer, you can also manually calculate the corresponding Up-vector and Latitude map by following Equations 1 and 2 in our paper.
Our code currently supports:
1) [Pinhole model](https://hedivision.github.io/Pinhole.html) [Hartley and Zisserman 2004] (Perspective Projection) 
```python
from perspective2d.utils.panocam import PanoCam
# define parameters
roll = 0
pitch = 20
vfov = 70
width = 640
height = 480
# get Up-vectors.
up = PanoCam.get_up(np.radians(vfov), width, height, np.radians(pitch), np.radians(roll))
# get Latitude.
lati = PanoCam.get_lat(np.radians(vfov), width, height, np.radians(pitch), np.radians(roll))
```
2) [Unified Spherical Model](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view) [Barreto 2006; Mei and Rives 2007] (Distortion). 
```python
xi = 0.5 # distortion parameter from Unified Spherical Model

x = -np.sin(np.radians(vfov/2))
z = np.sqrt(1 - x**2)
f_px_effective = -0.5*(width/2)*(xi+z)/x
crop, _, _, _, up, lat, xy_map = PanoCam.crop_distortion(equi_img,
                                             f=f_px_effective,
                                             xi=xi,
                                             H=height,
                                             W=width,
                                             az=yaw, # degrees
                                             el=-pitch,
                                             roll=-roll)
```

## Visualize Perspective Fields
We provide a one-line code to blend Perspective Fields onto input image.
```python
import matplotlib.pyplot as plt
from perspective2d.utils import draw_perspective_fields
# Draw up and lati on img. lati is in radians.
blend = draw_perspective_fields(img, up, lati)
# visualize with matplotlib
plt.imshow(blend)
plt.show()
```
Perspective Fields can serve as an easy visual check for correctness of the camera parameters.

- For example, we can visualize the Perspective Fields based on calibration results from this awesome [repo](https://github.com/dompm/spherical-distortion-dataset).


<p align="center">

![alt text](assets/distortion_vis.png)

- Left: We plot the perspective fields based on the numbers printed on the image, they look accurate😊;

- Mid: If we try a number that is 10% off (0.72*0.9=0.648), we see mismatch in Up directions at the top right corner;

- Right: If distortion is 20% off (0.72*0.8=0.576), the mismatch becomes more obvious.
</p>


Citation
--------
If you find this code useful, please consider citing:

```text
@inproceedings{jin2023perspective,
      title={Perspective Fields for Single Image Camera Calibration},
      author={Linyi Jin and Jianming Zhang and Yannick Hold-Geoffroy and Oliver Wang and Kevin Matzen and Matthew Sticha and David F. Fouhey},
      booktitle = {CVPR},
      year={2023}
}
```

Acknowledgment
--------------
This work was partially funded by the DARPA Machine Common Sense Program.
We thank authors from [A Deep Perceptual Measure for Lens and Camera Calibration](https://github.com/dompm/spherical-distortion-dataset) for releasing their code on Unified Spherical Model.
