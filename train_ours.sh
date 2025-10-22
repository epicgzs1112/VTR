#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
torchrun --nproc_per_node=1 --master_port=29691 train_on_voxel_VTR.py --AT_type ours --attack_model pix2vox --n_views 3 --interval_degree 30 --train_batch_size 8 --test_batch_size 196 --epochs_fintune 1 --ft_rate 0.001 --epochs 100

