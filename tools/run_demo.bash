# python demo/demo.py \
# --config-file ../jupyter-notebooks/models/paramnet_gsv_rpfpp.yaml \
# --input ../assets/cityscape.jpg \
# --output debug \
# --optimize \
# --noncenter \
# --net-init \
# --opts MODEL.WEIGHTS ../jupyter-notebooks/models/paramnet_gsv_rpfpp.pth


python demo/demo.py \
--config-file ../jupyter-notebooks/models/cvpr2023.yaml \
--input ../assets/imgs \
--output debug \
--opts MODEL.WEIGHTS ../jupyter-notebooks/models/cvpr2023.pth
