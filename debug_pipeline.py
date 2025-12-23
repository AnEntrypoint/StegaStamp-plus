#!/usr/bin/env python3
"""
Step-by-step debugging of the 256-bit training pipeline.
Traces data flow and checks for information loss.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 256
BATCH_SIZE = 2

print("="*80)
print("STEP-BY-STEP PIPELINE DEBUG")
print("="*80)

# ===== STEP 1: Load real training data =====
print("\n[STEP 1] Loading real training images...")
img_files = sorted(glob.glob('images_400x400/*.jpg'))[:5]
if not img_files:
    print("ERROR: No images found in images_400x400/. Creating synthetic data instead.")
    images_raw = np.random.uniform(0, 255, (BATCH_SIZE, HEIGHT, WIDTH, 3)).astype(np.uint8)
else:
    print(f"Found {len(img_files)} images")
    images_raw = []
    for img_file in img_files[:BATCH_SIZE]:
        img = tf.keras.preprocessing.image.load_img(img_file, target_size=(HEIGHT, WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)
        images_raw.append(img_array)
    images_raw = np.array(images_raw)

print(f"✓ Raw images shape: {images_raw.shape}, dtype: {images_raw.dtype}")
print(f"✓ Raw images range: [{images_raw.min()}, {images_raw.max()}]")

# ===== STEP 2: Normalize images =====
print("\n[STEP 2] Normalizing images...")
images = images_raw.astype(np.float32) / 255.0
images_normalized = images - 0.5  # Center around 0

print(f"✓ Images (float) shape: {images.shape}, dtype: {images.dtype}")
print(f"✓ Images range: [{images.min():.3f}, {images.max():.3f}]")
print(f"✓ Normalized images range: [{images_normalized.min():.3f}, {images_normalized.max():.3f}]")

# ===== STEP 3: Generate secrets =====
print("\n[STEP 3] Generating secrets...")
# Test 1: All zeros
secrets_zeros = np.zeros((BATCH_SIZE, SECRET_SIZE), dtype=np.float32)
# Test 2: All ones
secrets_ones = np.ones((BATCH_SIZE, SECRET_SIZE), dtype=np.float32)
# Test 3: First half zeros, second half ones
secrets_split = np.concatenate([
    np.zeros((BATCH_SIZE, SECRET_SIZE // 2), dtype=np.float32),
    np.ones((BATCH_SIZE, SECRET_SIZE // 2), dtype=np.float32)
], axis=1)
# Test 4: Random
secrets_random = np.random.binomial(1, 0.5, (BATCH_SIZE, SECRET_SIZE)).astype(np.float32)

print(f"✓ Secrets shape: {secrets_zeros.shape}")
print(f"✓ Secrets dtype: {secrets_zeros.dtype}")
print(f"✓ Secret patterns created:")
print(f"  - All zeros: sum={secrets_zeros.sum()}")
print(f"  - All ones: sum={secrets_ones.sum()}")
print(f"  - Split: sum={secrets_split.sum()}")
print(f"  - Random: sum={secrets_random.sum()}")

# ===== STEP 4: Build encoder =====
print("\n[STEP 4] Building encoder...")
class StegaStampEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.secret_dense = keras.layers.Dense(19200, activation='relu', kernel_initializer='he_normal')
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv4 = keras.layers.Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv5 = keras.layers.Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.up6 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = keras.layers.Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - 0.5
        image = image - 0.5
        secret = self.secret_dense(secret)
        secret = keras.layers.Reshape((80, 80, 3))(secret)
        secret_enlarged = keras.layers.UpSampling2D(size=(5, 5))(secret)
        x = keras.layers.Concatenate()([secret_enlarged, image])
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(keras.layers.UpSampling2D(size=(2, 2))(conv5))
        merge6 = keras.layers.Concatenate()([conv4, up6])
        conv6 = self.conv6(merge6)
        up7 = self.up7(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.Concatenate()([conv3, up7])
        conv7 = self.conv7(merge7)
        up8 = self.up8(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.Concatenate()([conv2, up8])
        conv8 = self.conv8(merge8)
        up9 = self.up9(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.Concatenate()([conv1, up9, x])
        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        residual = self.residual(conv10)
        return residual

encoder = StegaStampEncoder()
print("✓ Encoder created")

# ===== STEP 5: Test encoder with different secrets =====
print("\n[STEP 5] Testing encoder with different secrets...")
images_tf = tf.constant(images_normalized)

residual_zeros = encoder([tf.constant(secrets_zeros), images_tf], training=True)
residual_ones = encoder([tf.constant(secrets_ones), images_tf], training=True)
residual_split = encoder([tf.constant(secrets_split), images_tf], training=True)
residual_random = encoder([tf.constant(secrets_random), images_tf], training=True)

print(f"✓ All residuals have shape: {residual_zeros.shape}")
print(f"Residuals for different secrets:")
print(f"  - Zeros secret → residual sum: {tf.reduce_sum(tf.abs(residual_zeros)).numpy():.6f}, mean: {tf.reduce_mean(tf.abs(residual_zeros)).numpy():.6f}")
print(f"  - Ones secret  → residual sum: {tf.reduce_sum(tf.abs(residual_ones)).numpy():.6f}, mean: {tf.reduce_mean(tf.abs(residual_ones)).numpy():.6f}")
print(f"  - Split secret → residual sum: {tf.reduce_sum(tf.abs(residual_split)).numpy():.6f}, mean: {tf.reduce_mean(tf.abs(residual_split)).numpy():.6f}")
print(f"  - Random secret→ residual sum: {tf.reduce_sum(tf.abs(residual_random)).numpy():.6f}, mean: {tf.reduce_mean(tf.abs(residual_random)).numpy():.6f}")

# Check if residuals are different for different secrets
diff_zeros_ones = tf.reduce_mean(tf.abs(residual_zeros - residual_ones)).numpy()
diff_zeros_split = tf.reduce_mean(tf.abs(residual_zeros - residual_split)).numpy()
diff_zeros_random = tf.reduce_mean(tf.abs(residual_zeros - residual_random)).numpy()

print(f"\nDifferences between residuals:")
print(f"  - Zeros vs Ones:  {diff_zeros_ones:.8f}")
print(f"  - Zeros vs Split: {diff_zeros_split:.8f}")
print(f"  - Zeros vs Random:{diff_zeros_random:.8f}")

if diff_zeros_ones < 1e-4 and diff_zeros_split < 1e-4 and diff_zeros_random < 1e-4:
    print("  ⚠️  WARNING: Encoder producing nearly identical residuals for ALL secrets!")
    print("  ⚠️  This is the root cause - encoder is not differentiating based on secrets")

# ===== STEP 6: Build decoder =====
print("\n[STEP 6] Building decoder...")
class StegaStampDecoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.decoder = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(SECRET_SIZE)
        ])

    def call(self, image):
        image = image - 0.5
        return self.decoder(image)

decoder = StegaStampDecoder()
print("✓ Decoder created")

# ===== STEP 7: Test decoder on encoded images =====
print("\n[STEP 7] Testing decoder on encoded images...")
encoded_zeros = images_normalized + residual_zeros.numpy()
encoded_ones = images_normalized + residual_ones.numpy()
encoded_split = images_normalized + residual_split.numpy()
encoded_random = images_normalized + residual_random.numpy()

decoded_zeros = decoder(tf.constant(encoded_zeros), training=True).numpy()
decoded_ones = decoder(tf.constant(encoded_ones), training=True).numpy()
decoded_split = decoder(tf.constant(encoded_split), training=True).numpy()
decoded_random = decoder(tf.constant(encoded_random), training=True).numpy()

print(f"✓ All decoded have shape: {decoded_zeros.shape}")
print(f"\nDecoded output statistics:")
print(f"  - From zeros secret:  mean={decoded_zeros.mean():.4f}, std={decoded_zeros.std():.4f}")
print(f"  - From ones secret:   mean={decoded_ones.mean():.4f}, std={decoded_ones.std():.4f}")
print(f"  - From split secret:  mean={decoded_split.mean():.4f}, std={decoded_split.std():.4f}")
print(f"  - From random secret: mean={decoded_random.mean():.4f}, std={decoded_random.std():.4f}")

# Compare decoded outputs
diff_dec_zeros_ones = np.mean(np.abs(decoded_zeros - decoded_ones))
diff_dec_zeros_split = np.mean(np.abs(decoded_zeros - decoded_split))
diff_dec_zeros_random = np.mean(np.abs(decoded_zeros - decoded_random))

print(f"\nDecoded output differences:")
print(f"  - Zeros vs Ones:  {diff_dec_zeros_ones:.8f}")
print(f"  - Zeros vs Split: {diff_dec_zeros_split:.8f}")
print(f"  - Zeros vs Random:{diff_dec_zeros_random:.8f}")

if diff_dec_zeros_ones < 0.01 and diff_dec_zeros_split < 0.01 and diff_dec_zeros_random < 0.01:
    print("  ⚠️  WARNING: Decoder producing nearly identical outputs regardless of residuals!")
    print("  ⚠️  Decoder cannot distinguish between different encoded secrets")

# ===== STEP 8: Compute losses =====
print("\n[STEP 8] Computing losses...")
secret_loss_zeros = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets_zeros, logits=decoded_zeros))
secret_loss_ones = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets_ones, logits=decoded_ones))
secret_loss_split = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets_split, logits=decoded_split))
secret_loss_random = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets_random, logits=decoded_random))

print(f"Secret loss for different cases:")
print(f"  - All zeros:  {secret_loss_zeros.numpy():.6f} (target: 0.0 for perfect reconstruction)")
print(f"  - All ones:   {secret_loss_ones.numpy():.6f} (target: 0.0 for perfect reconstruction)")
print(f"  - Split:      {secret_loss_split.numpy():.6f} (target: 0.69 for random)")
print(f"  - Random:     {secret_loss_random.numpy():.6f} (target: 0.69 for random)")

random_baseline = np.log(2)  # 0.6931
print(f"\nRandom baseline for binary classification: {random_baseline:.6f}")
print(f"All losses near baseline: {all([l.numpy() > 0.69 for l in [secret_loss_zeros, secret_loss_ones, secret_loss_split, secret_loss_random]])}")

# ===== SUMMARY =====
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
if diff_zeros_ones < 1e-4:
    print("❌ CRITICAL: Encoder is not responding to secret differences")
    print("   The encoder produces identical residuals regardless of input secrets")
    print("   This explains why loss stays at random baseline 0.6931")
    print("\n   Root cause: Encoder secret pathway may be broken or ignored")
elif diff_dec_zeros_ones < 0.01:
    print("❌ CRITICAL: Decoder cannot extract information from residuals")
    print("   The decoder produces identical outputs regardless of residuals")
    print("   This explains why decoding always fails")
    print("\n   Root cause: Residuals don't contain encodable signal")
else:
    print("✓ Pipeline appears to be functioning")
    print("   Encoder responds to secrets and decoder can extract them")
