#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

HEIGHT, WIDTH = 256, 256
SECRET_SIZE = 256
BATCH_SIZE = 4
NUM_TEST_SAMPLES = 50

print("Loading final trained models...")
try:
    encoder = keras.models.load_model('encoder.keras')
    decoder = keras.models.load_model('decoder.keras')
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    exit(1)

print(f"\nGenerating {NUM_TEST_SAMPLES} test samples...")
test_images = np.random.rand(NUM_TEST_SAMPLES, HEIGHT, WIDTH, 3).astype(np.float32)
test_secrets_random = (np.random.binomial(1, 0.5, (NUM_TEST_SAMPLES, SECRET_SIZE))).astype(np.float32)

print("Testing encoder/decoder pipeline...")
batch_images = tf.constant(test_images[:BATCH_SIZE])
batch_secrets = tf.constant(test_secrets_random[:BATCH_SIZE])

encoded = encoder([batch_images, batch_secrets], training=False)
decoded = decoder(encoded, training=False)

predictions = (decoded.numpy() > 0.5).astype(int)
ground_truth = (test_secrets_random[:BATCH_SIZE] > 0.5).astype(int)
accuracy = np.mean(predictions == ground_truth) * 100

print(f"\nTest Results (Random 256-bit Secrets):")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Message accuracy: {accuracy:.2f}%")
print(f"  Encoded image MSE: {np.mean((encoded.numpy() - batch_images.numpy())**2):.6f}")

sample_pred = predictions[0]
sample_true = ground_truth[0]
bit_errors = np.sum(sample_pred != sample_true)
print(f"\nSample bit accuracy: {100 - (bit_errors/256*100):.2f}% ({256-bit_errors}/256 correct)")
print(f"Sample bit errors: {bit_errors}/256")

print("\n✓ Final models tested successfully")
