#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if len(sys.argv) < 3:
    print("Usage: python3 infer_256bit.py <checkpoint_step> <image_path>")
    print("Example: python3 infer_256bit.py 10000 test_image.jpg")
    sys.exit(1)

checkpoint_step = sys.argv[1]
image_path = sys.argv[2]

decoder_path = f'decoder_256bit_step_{checkpoint_step}.keras'

if not os.path.exists(decoder_path):
    print(f"Error: {decoder_path} not found")
    print("Available checkpoints:")
    for f in sorted(glob.glob('decoder_256bit_step_*.keras')):
        print(f"  {f}")
    sys.exit(1)

print(f"Loading decoder from checkpoint {checkpoint_step}...", flush=True)
decoder = keras.models.load_model(decoder_path)

print(f"Loading image from {image_path}...", flush=True)
img = Image.open(image_path).convert("RGB")
img = ImageOps.fit(img, (400, 400))
img_array = np.array(img, dtype=np.float32) / 255.0
img_tensor = tf.constant(img_array[np.newaxis, ...])

print("Extracting 256-bit secret...", flush=True)
logits = decoder(img_tensor, training=False)
probs = tf.nn.sigmoid(logits).numpy()[0]
bits = (probs > 0.5).astype(int)

confidence = np.abs(probs - 0.5) + 0.5
mean_confidence = np.mean(confidence)

print(f"\n256-bit extraction from checkpoint {checkpoint_step}:")
print(f"Bits (first 32): {bits[:32]}")
print(f"Bits (last 32): {bits[-32:]}")
print(f"Mean confidence: {mean_confidence:.4f}")
print(f"Min/Max probability: [{np.min(probs):.4f}, {np.max(probs):.4f}]")
print(f"\nFull 256-bit secret: {''.join(map(str, bits))}")
