#!/bin/bash

echo "=================================="
echo "StegaStamp WebGPU - Quick Setup"
echo "=================================="
echo ""
echo "This script will:"
echo "  1. Train models using RTX 3060"
echo "  2. Convert to ONNX"
echo "  3. Start web server"
echo ""

set -e

# Step 1: Ensure dependencies
echo "[1/4] Installing dependencies..."
pip install --break-system-packages \
    'tensorflow>=2.16' \
    'tf2onnx' \
    'onnx' \
    'numpy' \
    'opencv-python' \
    'pillow' \
    2>&1 | grep -E "(Successfully|Requirement|Installing)"

# Step 2: Train models
echo ""
echo "[2/4] Training models (GPU)..."
python3 train_local.py

# Step 3: Convert to ONNX
echo ""
echo "[3/4] Converting to ONNX..."
python3 scripts/convert-to-onnx.py

# Step 4: Start web server
echo ""
echo "[4/4] Starting web server..."
npm run dev

echo ""
echo "=================================="
echo "✅ Ready! Open http://localhost:5173"
echo "=================================="
