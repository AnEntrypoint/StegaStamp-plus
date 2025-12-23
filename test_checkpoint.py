#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from PIL import Image, ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_PATH = './data/train'
SECRET_SIZE = 256
BATCH_SIZE = 4
NUM_TEST_BATCHES = 10

if len(sys.argv) < 2:
    print("Usage: python3 test_checkpoint.py <checkpoint_step>")
    print("Example: python3 test_checkpoint.py 10000")
    sys.exit(1)

checkpoint_step = sys.argv[1]
encoder_path = f'encoder_256bit_step_{checkpoint_step}.keras'
decoder_path = f'decoder_256bit_step_{checkpoint_step}.keras'

if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
    print(f"Error: Checkpoint {checkpoint_step} not found")
    print("Available checkpoints:")
    for f in sorted(glob.glob('encoder_256bit_step_*.keras')):
        step = f.split('_')[-1].replace('.keras', '')
        print(f"  Step {step}")
    sys.exit(1)

print(f"Loading checkpoint {checkpoint_step}...", flush=True)
encoder = keras.models.load_model(encoder_path)
decoder = keras.models.load_model(decoder_path)

print(f"Loading {NUM_TEST_BATCHES} test batches...", flush=True)
img_files = (glob.glob(os.path.join(TRAIN_PATH, '*.jpg')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.png')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.webp')))

if not img_files:
    print(f"Error: No images found in {TRAIN_PATH}")
    sys.exit(1)

accuracies = []
secret_losses = []
l2_losses = []

for batch_idx in range(NUM_TEST_BATCHES):
    batch_cover = []
    batch_secret = []

    for _ in range(BATCH_SIZE):
        try:
            img_path = np.random.choice(img_files)
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, (400, 400))
            img = np.array(img, dtype=np.float32) / 255.0
        except:
            img = np.zeros((400, 400, 3), dtype=np.float32)
        batch_cover.append(img)

        secret = np.random.binomial(1, 0.5, SECRET_SIZE).astype(np.float32)
        batch_secret.append(secret)

    images = tf.constant(np.array(batch_cover))
    secrets = tf.constant(np.array(batch_secret))

    residual = encoder([secrets, images], training=False)
    encoded = images + residual
    decoded = decoder(encoded, training=False)

    secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
    l2_loss = tf.reduce_mean(tf.square(residual))

    probs = tf.nn.sigmoid(decoded).numpy()
    predicted = (probs > 0.5).astype(int)
    accuracy = np.mean(predicted == batch_secret)

    accuracies.append(accuracy)
    secret_losses.append(float(secret_loss))
    l2_losses.append(float(l2_loss))

    print(f"Batch {batch_idx+1}/{NUM_TEST_BATCHES} | Accuracy: {accuracy:.4f} | SecretLoss: {float(secret_loss):.4f} | L2Loss: {float(l2_loss):.4f}", flush=True)

mean_acc = np.mean(accuracies)
mean_secret_loss = np.mean(secret_losses)
mean_l2_loss = np.mean(l2_losses)

print(f"\n{'='*70}")
print(f"CHECKPOINT {checkpoint_step} EVALUATION")
print(f"{'='*70}")
print(f"Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"Target: >50% (random baseline) for learning signal")
print(f"Mean Secret Loss: {mean_secret_loss:.4f}")
print(f"Mean L2 Loss: {mean_l2_loss:.4f}")
print(f"{'='*70}")

if mean_acc > 0.5:
    print("✓ PASS: Model learning detected (accuracy > 50%)")
else:
    print("✗ FAIL: No learning detected (accuracy ≤ 50%)")

if mean_secret_loss < 0.69:
    print("✓ PASS: Secret loss below random baseline (0.693)")
else:
    print("✗ FAIL: Secret loss at or above random baseline")
