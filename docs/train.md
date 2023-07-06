# Datasets

We used Google street view dataset to train the ParamNet in our paper Table 3, 4.
Download GSV datasets:
```bash
wget https://www.dropbox.com/s/plcmcza8vfmmpkm/google_street_view_191210.tar
wget https://www.dropbox.com/s/9se3lrpljd59cod/gsv_test_crop_uniform.tar
```
Extract the dataset under `perspectiveField/datasets`.


Note that we used images from [360cities](https://www.360cities.net) to train the PerspectiveNet in Table 1.


# Training PerspectiveNet + ParamNet
Download initial weights from segformer.b3.512x512.ade.160k.pth:
```bash
wget https://www.dropbox.com/s/0axxpfga265gq3o/ade_pretrained.pth
```
Place it under `perspectiveField/init_model_weights`.


- We first trained PerspectiveNet:
```bash
python -W ignore train.py \
--config-file configs/config-mix-gsv-regress.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:$((RANDOM +10000)) \
OUTPUT_DIR "./exp/step01-gsv-perspective-pretrain" \
SOLVER.IMS_PER_BATCH 64
```

- Then we trained the ParamNet, you can download the model from the previous step here:
```bash
wget https://www.dropbox.com/s/c9199n5lmy30tob/gsv_persnet_pretrain.pth
```

- To train the ParamNet to predict roll, pitch and fov:
```bash
python -W ignore train.py \
--config-file configs/config-gsv-rpf.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:$((RANDOM +10000)) \
OUTPUT_DIR "./exp/step02-gsv-paramnet-rpf"
```

- To train the ParamNet to predict roll, pitch, fov, and principal point:
```bash
python -W ignore train.py \
--config-file configs/config-gsv-rpfpp.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:$((RANDOM +10000)) \
OUTPUT_DIR "./exp/step02-gsv-paramnet-rpfpp"
```
