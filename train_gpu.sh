#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

echo "Environment:"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"
echo ""

python3 train.py
