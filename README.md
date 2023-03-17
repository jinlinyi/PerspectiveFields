Perspective Fields for Single Image Camera Calibration
================================================================

<h4>
Linyi Jin, Jianming Zhang, Yannick Hold-Geoffroy, Oliver Wang, Kevin Matzen, Matthew Sticha, David Fouhey
</br>
<span style="font-size: 14pt; color: #555555">
University of Michigan, Adobe Research
</span>
</br>
CVPR 2023
</h4>
<hr>

![alt text](assets/teaser-field.jpg)
We propose Perspective Fields as a representation that models the local perspective properties of an image. Perspective Fields contain per-pixel information about the camera view, parameterized as an up vector and a latitude value.

This repo contains code for our [paper][0]. 


<img height="150" alt="swiping-1" src="assets/swiping-1.gif"> <img height="150" alt="swiping-2" src="assets/swiping-2.gif"> <img height="150" alt="swiping-3" src="assets/swiping-3.gif"> <img height="150" alt="swiping-4" src="assets/swiping-4.gif">



Usage Instructions
------------------

1. [Setup your environment][1]
2. [Compute perspective fields based on pin-hole camera parameters][2]
3. [Inference on a single image][3]
4. [Get camera parameters from Perspective Fields (TODO)][6]
5. [Train (TODO)][4]
6. [Test (TODO)][5]

[0]: https://arxiv.org/abs/2212.03239
[1]: ./docs/environment.md
[2]: ./jupyter-notebooks/camera2perspective.ipynb
[3]: ./docs/jupyter-notebooks/predict_perspective_fields.ipynb
[4]: README.md
[5]: README.md
[6]: README.md


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
We thank Geoffrey Oxholm for the help with
Upright, and Aaron Hertzmann, Scott Cohen, 
Ang Cao, Dandan Shan, Mohamed El Banani, Sarah Jabbour, Shengyi Qian for discussions. 
