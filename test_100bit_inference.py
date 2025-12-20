#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SECRET_SIZE = 100
IMG_SIZE = 400

if len(sys.argv) < 2:
    print("Usage: python3 test_100bit_inference.py <step>", flush=True)
    print("Example: python3 test_100bit_inference.py 10000", flush=True)
    sys.exit(1)

step = int(sys.argv[1])
encoder_path = f'encoder_100bit_step_{step}.keras'
decoder_path = f'decoder_100bit_step_{step}.keras'

if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
    print(f"✗ Checkpoint not found: {encoder_path} or {decoder_path}", flush=True)
    sys.exit(1)

print(f"Testing encoder/decoder at step {step}...", flush=True)
encoder = keras.models.load_model(encoder_path)
decoder = keras.models.load_model(decoder_path)
print("✓ Models loaded", flush=True)

np.random.seed(42)
accuracies = []

for test_idx in range(20):
    secret = np.random.binomial(1, 0.5, (1, SECRET_SIZE)).astype(np.float32)
    image = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    residual = encoder([tf.constant(secret), tf.constant(image)], training=False)
    encoded = image + residual

    decoded = decoder(encoded, training=False)
    pred_bits = (decoded.numpy()[0] > 0.0).astype(int)
    secret_bits = (secret[0] > 0.5).astype(int)

    acc = np.mean(pred_bits == secret_bits)
    accuracies.append(acc)

avg_acc = np.mean(accuracies)
print(f"Average accuracy: {avg_acc:.1%}", flush=True)
print(f"Random baseline: 50.0%", flush=True)

if avg_acc > 0.5:
    print(f"✓ Model is learning! ({(avg_acc-0.5)*100:.1f}% above random)", flush=True)
else:
    print(f"✗ Model not learning (at random baseline)", flush=True)
