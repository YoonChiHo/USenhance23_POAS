CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
--dataroot ../datasets/low2high_v2 \
--name miccai_chall_L2H_pixelSh_MTL_with_pretrained_400 \
--gpu_ids 0,1 \
--input_nc 1 \
--output_nc 1 \
--batch_size 24 \
--phase train \
--is_mtl

CUDA_VISIBLE_DEVICES=0,1 \
python test.py \
--results_dir ../results/miccai_chall_L2H_pixelSh_MTL_with_pretrained_400 \
--name miccai_chall_L2H_pixelSh_MTL_with_pretrained_400 \
--model_suffix _A \
--dataset_mode unaligned \
--gpu_ids 0,1 \
--input_nc 1 \
--output_nc 1 \
--batch_size 24 \
--phase test \
--dataroot ../datasets/low2high_v2 \
--is_mtl 
