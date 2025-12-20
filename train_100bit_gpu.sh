#!/bin/bash
export LD_LIBRARY_PATH="/home/user/diffusers/pixel_art_venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.6"
cd /home/user/StegaStamp-plus
python3 train_100bit.py
