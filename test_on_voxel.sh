#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29522 train_on_voxel_VTR.py --attack_model lrgt --n_views 3 --test --test_batch_size 1 --test_ckpt /mnt/d/zhouhang/zero123/lrgt/output/checkpoints/AT_30_degree_ours/checkpoint_best.pth --rot
