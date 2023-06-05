CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--config-file configs/config-edina-rpfpp.yaml \
--num-gpus 1 \
--dist-url tcp://127.0.0.1:$((RANDOM +10000)) \