#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29520 train_on_voxel_VTR.py --attack_model pix2vox --n_views 3 --test --test_batch_size 1 --test_ckpt /home/ubuntu/zhouhang/zero123-main/zero123/pix2vox/output/checkpoints/AT_30_degree_10ours/checkpoint_best.pth --rot