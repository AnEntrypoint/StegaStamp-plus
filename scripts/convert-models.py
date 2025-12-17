#!/usr/bin/env python3
import os
import subprocess
import sys

def convert_tf_to_onnx(tf_model_path, onnx_output_path):
    print(f"Converting {tf_model_path} → {onnx_output_path}")
    try:
        cmd = [
            "python",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            tf_model_path,
            "--output",
            onnx_output_path,
            "--opset",
            "13",
            "--verbose"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Successfully converted to {onnx_output_path}")
            return True
        else:
            print(f"✗ Conversion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    models_dir = "models"

    conversions = [
        (f"{models_dir}/saved_models/stegastamp_pretrained", "public/models/encoder.onnx"),
        (f"{models_dir}/detector_models/stegastamp_detector", "public/models/detector.onnx"),
    ]

    os.makedirs("public/models", exist_ok=True)

    for tf_path, onnx_path in conversions:
        if os.path.exists(tf_path):
            convert_tf_to_onnx(tf_path, onnx_path)
        else:
            print(f"⚠ Warning: {tf_path} not found")

    print("\nModel conversion complete!")
