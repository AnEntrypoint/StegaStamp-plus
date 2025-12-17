#!/usr/bin/env python3
"""
Minimal model generator - creates dummy ONNX models without TensorFlow
This allows testing the web frontend while TensorFlow installs
"""

import os
import struct
import numpy as np

def create_minimal_onnx():
    """Create minimal valid ONNX models for testing"""

    os.makedirs("public/models", exist_ok=True)

    print("Creating minimal ONNX models for web frontend testing...")

    encoder_bytes = bytes([
        0x08, 0x03, 0x12, 0x07, 0x0a, 0x04, 0x6f, 0x72, 0x74, 0x6d, 0x10, 0x0d,
        0x1a, 0x0b, 0x08, 0x01, 0x12, 0x07, 0x0a, 0x04, 0x6f, 0x72, 0x74, 0x6d,
    ] + [0] * 1000)

    decoder_bytes = bytes([
        0x08, 0x03, 0x12, 0x07, 0x0a, 0x04, 0x6f, 0x72, 0x74, 0x6d, 0x10, 0x0d,
        0x1a, 0x0b, 0x08, 0x01, 0x12, 0x07, 0x0a, 0x04, 0x6f, 0x72, 0x74, 0x6d,
    ] + [0] * 1000)

    with open("public/models/encoder.onnx", "wb") as f:
        f.write(encoder_bytes)

    with open("public/models/decoder.onnx", "wb") as f:
        f.write(decoder_bytes)

    print("✅ Models created:")
    print(f"   encoder.onnx: {len(encoder_bytes)} bytes")
    print(f"   decoder.onnx: {len(decoder_bytes)} bytes")

if __name__ == "__main__":
    create_minimal_onnx()
    print("\nNote: These are placeholder models. For real models, run:")
    print("  1. Wait for TensorFlow to install")
    print("  2. python3 train_local.py")
    print("  3. python3 scripts/convert-to-onnx.py")
