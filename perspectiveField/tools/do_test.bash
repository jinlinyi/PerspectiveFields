GPU=0
echo "GPU" $GPU

# # Test offcenter crop
# CUDA_VISIBLE_DEVICES=0, python demo/offcenter_crop.py \
# --config-file configs/persformer/config-up-only.yaml#configs/persformer/config-lati-only-cls.yaml \
# --output ./debug \
# --dataset cities360_test \
# --opts MODEL.WEIGHTS /home/code-base/user_space/exp/e08_persformer/e07_up_v3/model_0059999.pth#/home/code-base/user_space/exp/e08_persformer/e09_lati_v3_cls/model_0031999.pth

# echo "Eval model"
# CUDA_VISIBLE_DEVICES=$GPU python tools/train.py \
# --config-file /home/code-base/user_space/exp/e08_persformer/e13_up_v4/config.yaml \
# --eval-only \
# MODEL.WEIGHTS /home/code-base/user_space/exp/e08_persformer/e13_up_v4/model_final.pth \
# DATASETS.TEST "('stanford2d3d_test', 'tartanair_test')" \
# OUTPUT_DIR ./debug

# echo "Test uncrop"
# dataset="objectron_test_crop"
# echo $dataset
# CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_warp.py \
# --config-file /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/config.yaml#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/config.yaml \
# --output ./debug \
# --dataset $dataset \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/object_centric/e05_up_72_coco_pseudo_randomcrop/model_0005999.pth#/home/jinlinyi/exp/densefield/object_centric/e04_lati_cls_180/model_0045999.pth
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/model_0059999.pth#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/model_0021999.pth

# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/object_centric/e05_up_72_coco_pseudo_randomcrop/model_0005999.pth#/home/jinlinyi/exp/densefield/object_centric/e04_lati_cls_180/model_0045999.pth
# for dataset in stanford2d3d_test stanford2d3d_test_crop stanford2d3d_test_warp tartanair_test tartanair_test_crop tartanair_test_warp objectron_test_crop objectron_test_crop_mask
# do
#     echo $dataset
#     CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_warp.py \
#     --dataset $dataset \
#     --expname ours \
#     --output /home/jinlinyi/exp/densefield/e37_360_regress_ade/lr-3/model_final_test \
#     --config-file /home/jinlinyi/exp/densefield/e37_360_regress_ade/lr-3/config.yaml \
#     --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e37_360_regress_ade/lr-3/model_final.pth
# done

# for dataset in stanford2d3d_test stanford2d3d_test_crop stanford2d3d_test_warp tartanair_test tartanair_test_crop tartanair_test_warp objectron_test_crop objectron_test_crop_mask
# do
#     echo $dataset
#     CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_warp.py \
#     --dataset $dataset \
#     --expname ours \
#     --output /home/jinlinyi/exp/densefield/e36_recover_rpfpp_v3_small/regression-all-lr2-rebalanced/dense_model_0078999 \
#     --config-file /home/jinlinyi/exp/densefield/e32_gsv_regress_ade/e04-lr-3-rebalanced/config.yaml \
#     --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e36_recover_rpfpp_v3_small/regression-all-lr2-rebalanced/model_0078999.pth
# done
# for dataset in objectron_test_crop objectron_test_crop_mask
# for dataset in gsv_test_crop
# for dataset in stanford2d3d_test stanford2d3d_test_crop stanford2d3d_test_warp tartanair_test tartanair_test_crop tartanair_test_warp objectron_test_crop objectron_test_crop_mask
# do
#     echo $dataset
#     CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_warp.py \
#     --dataset $dataset \
#     --expname ours \
#     --output /home/jinlinyi/exp/densefield/e08_persformer/e15_latiup_mixed_v3/test \
#     --config-file /home/jinlinyi/exp/densefield/e08_persformer/e15_latiup_mixed_v3/config.yaml \
#     --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e08_persformer/e15_latiup_mixed_v3/model_final.pth
# done
# for dataset in stanford2d3d_test stanford2d3d_test_crop stanford2d3d_test_warp tartanair_test tartanair_test_crop tartanair_test_warp objectron_test_crop objectron_test_crop_mask
for dataset in edina_test_crop_uniform
do
    echo $dataset
    CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_warp.py \
    --dataset $dataset \
    --expname ours \
    --output /home/msticha/exps/edina_test00 \
    --config-file /home/msticha/workspace/PerspectiveFields/perspectiveField/configs/config-edina-rpfpp.yaml \
    --opts MODEL.WEIGHTS /home/msticha/exps/e00_edina_pp/model_final.pth
done

# echo "Test camera parameter - GSV cropped - ablation no loss"
CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_param_network.py \
--config-file /home/msticha/workspace/PerspectiveFields/perspectiveField/configs/config-edina-rpfpp.yaml \
--output /home/msticha/exps/edina_test00 \
--dataset edina_test_crop_uniform  \
--opts MODEL.WEIGHTS /home/msticha/exps/e00_edina_pp/model_final.pth

# for dataset in gsv_test gsv_test_crop
# do
#     echo $dataset
#     CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_warp.py \
#     --dataset $dataset \
#     --expname ours_fpn \
#     --output /Pool1/users/jinlinyi/exp/densefield/e28_eccv_rebuttal \
#     --config-file configs/mixed-fpn/mixed_config.yaml \
#     --opts MODEL.WEIGHTS /Pool1/users/jinlinyi/exp/densefield/e28_fpn/e00_debug/model_0073999.pth
    

#     # --config-file /Pool1/users/jinlinyi/exp/densefield/e27_gsv/e00_debug/config.yaml \
#     # --opts MODEL.WEIGHTS /Pool1/users/jinlinyi/exp/densefield/e27_gsv/e00_debug/model_final.pth


#     # --config-file /home/jinlinyi/exp/densefield/e25_latiup/e02_lr01/config.yaml \
#     # --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e25_latiup/e02_lr01/model_0079999.pth
    
#     # --expname ours \
#     # --config-file /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/config.yaml#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/config.yaml \
#     # --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/model_0059999.pth#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/model_0021999.pth
   
#     # --expname ours_coco \
#     # --config-file /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/config.yaml#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/config.yaml \
#     # --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/object_centric/e07_up_72_coco_pseudo_mask/model_0016999.pth#/home/jinlinyi/exp/densefield/object_centric/e08_lati_cls_180_pseudo_mask/model_0006499.pth
    
    
     
# done

# echo "Test crop"
# dataset="tartanair_test_crop"
# echo $dataset
# CUDA_VISIBLE_DEVICES=$GPU, python demo/test_warp.py \
# --config-file configs/persformer/config-up-only.yaml#/home/code-base/user_space/exp/e08_persformer/e10_lati_v4_cls_180/config.yaml \
# --output ./debug \
# --dataset $dataset \
# --opts MODEL.WEIGHTS /home/code-base/user_space/exp/e08_persformer/e07_up_v3/model_0059999.pth#/home/code-base/user_space/exp/e08_persformer/e10_lati_v4_cls_180/model_0021999.pth

# echo "Test warp"
# dataset="tartanair_test_warp"
# echo $dataset
# CUDA_VISIBLE_DEVICES=$GPU, python demo/test_warp.py \
# --config-file /home/code-base/user_space/exp/e08_persformer/e13_up_v4/config.yaml#/home/code-base/user_space/exp/e08_persformer/e10_lati_v4_cls_180/config.yaml \
# --output ./debug \
# --dataset $dataset \
# --opts MODEL.WEIGHTS /home/code-base/user_space/exp/e08_persformer/e07_up_v3/model_0059999.pth#/home/code-base/user_space/exp/e08_persformer/e10_lati_v4_cls_180/model_0021999.pth

# echo "Test camera parameter"
# CUDA_VISIBLE_DEVICES=1, python demo/test.py \
# --config-file configs/persformer/config-up-only.yaml#/home/code-base/user_space/exp/e08_persformer/e10_lati_v4_cls_180/config.yaml \
# --output ./debug \
# --dataset tartanair_test \
# --opts MODEL.WEIGHTS /home/code-base/user_space/exp/e08_persformer/e07_up_v3/model_0059999.pth#/home/code-base/user_space/exp/e08_persformer/e10_lati_v4_cls_180/model_0021999.pth

# echo "Test camera parameter"
# CUDA_VISIBLE_DEVICES=$GPU, python demo/test.py \
# --config-file /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/config.yaml#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/config.yaml \
# --output ./debug \
# --dataset stanford2d3d_test_crop \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e08_persformer/e07_up_v3/model_0059999.pth#/home/jinlinyi/exp/densefield/e08_persformer/e10_lati_v4_cls_180/model_0021999.pth

# echo "Test camera parameter"
# CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_param_network.py \
# --config-file /home/jinlinyi/exp/densefield/e33_recover_rpf/all-adepretrained-lr-2/config.yaml \
# --output ./debug \
# --dataset gsv_test \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e33_recover_rpf/all-adepretrained-lr-2/model_final.pth

# echo "Test camera parameter - GSV cropped"
# CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_param_network.py \
# --config-file /home/jinlinyi/exp/densefield/e36_recover_rpfpp_v3_small/regression-all-lr2-rebalanced/config.yaml \
# --output ./debug \
# --dataset gsv_test_crop_uniform  \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e36_recover_rpfpp_v3_small/regression-all-lr2-rebalanced/model_0079499.pth

# echo "Test camera parameter - GSV cropped - ablation no loss"
# CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_param_network.py \
# --config-file /home/jinlinyi/exp/densefield/e36_recover_rpfpp_v3_small/regression-baseline/config.yaml \
# --output ./debug \
# --dataset gsv_test_crop_uniform  \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e36_recover_rpfpp_v3_small/regression-baseline/model_0078499.pth

# echo "Test camera parameter - GSV cropped - ablation train on non-crop"
# CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_param_network.py \
# --config-file /home/jinlinyi/exp/densefield/e33_recover_rpf/all-adepretrained-lr-2/config.yaml \
# --output ./debug \
# --dataset gsv_test_crop_uniform  \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e33_recover_rpf/all-adepretrained-lr-2/model_final.pth

# echo "Test camera parameter - GSV cropped - ablation standalone"
# CUDA_VISIBLE_DEVICES=$GPU, python -W ignore demo/test_param_network.py \
# --config-file /home/jinlinyi/exp/densefield/e41_gsv_regress_standalone/e04_lr-3/config.yaml \
# --output ./debug \
# --dataset gsv_test_crop_uniform  \
# --opts MODEL.WEIGHTS /home/jinlinyi/exp/densefield/e41_gsv_regress_standalone/e04_lr-3/model_final.pth

# CUDA_VISIBLE_DEVICES=1,2, python demo/test.py \
# --config-file configs/persformer/config-latiup-mix.yaml \
# --output ./debug \
# --dataset hypersim_test \
# --opts MODEL.WEIGHTS /home/code-base/user_space/exp/e08_persformer/e02_latiup_mixed_v2/model_0069999.pth