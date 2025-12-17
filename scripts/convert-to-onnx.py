#!/usr/bin/env python3
"""Convert trained TensorFlow SavedModels to ONNX for browser"""

import os
import sys
import subprocess

print("="*80)
print("Converting TensorFlow Models to ONNX")
print("="*80)

models_to_convert = [
    ("models/saved_models/stegastamp_pretrained", "public/models/encoder.onnx", "encoder"),
    ("models/saved_models/decoder_model", "public/models/decoder.onnx", "decoder"),
]

os.makedirs("public/models", exist_ok=True)

for tf_path, onnx_path, name in models_to_convert:
    if not os.path.exists(tf_path):
        print(f"\n❌ {name}: TensorFlow model not found at {tf_path}")
        continue

    print(f"\n[*] Converting {name}...")
    print(f"    From: {tf_path}")
    print(f"    To:   {onnx_path}")

    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", tf_path,
        "--output", onnx_path,
        "--opset", "13",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            file_size = os.path.getsize(onnx_path) / (1024*1024)
            print(f"    ✅ Success ({file_size:.1f}MB)")
        else:
            print(f"    ❌ Failed: {result.stderr}")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print("\n" + "="*80)
print("Conversion complete!")
print("="*80)
print("\nModels ready for browser:")
for _, path, name in models_to_convert:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"  ✓ {path} ({size:.1f}MB)")

print("\nNext: npm run dev")
print("="*80)
