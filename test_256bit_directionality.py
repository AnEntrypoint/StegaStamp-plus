#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from PIL import Image, ImageOps

print("="*70)
print("256-BIT TRAINING DIRECTIONALITY TEST")
print("="*70)
print()

TRAIN_PATH = './data/train'
HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 256
BATCH_SIZE = 4
TEST_STEPS = 200

print(f"Test configuration: {TEST_STEPS} training steps (256-bit scaled)")
print(f"Phase 1 behavior: L2 loss ramps 0→0.5")
print()

img_files = (glob.glob(os.path.join(TRAIN_PATH, '*.jpg')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.png')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.webp')))

if not img_files:
    print("✗ ERROR: No training images found!")
    exit(1)

def get_img_batch(batch_size=BATCH_SIZE):
    batch_cover = []
    batch_secret = []

    for _ in range(batch_size):
        try:
            img_path = np.random.choice(img_files)
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, (HEIGHT, WIDTH))
            img = np.array(img, dtype=np.float32) / 255.0
        except:
            img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        batch_cover.append(img)

        secret = np.random.binomial(1, 0.5, SECRET_SIZE).astype(np.float32)
        batch_secret.append(secret)

    return np.array(batch_cover), np.array(batch_secret)

class StegaStampEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.secret_dense = keras.layers.Dense(16384, activation='relu', kernel_initializer='he_normal')
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
        secret = keras.layers.Reshape((64, 64, 4))(secret)
        secret = keras.layers.Conv2D(3, 1, activation='relu', padding='same')(secret)
        secret_enlarged = keras.layers.UpSampling2D(size=(6, 6))(secret)

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

print("Building models...", flush=True)
encoder = StegaStampEncoder()
decoder = StegaStampDecoder()

enc_opt = keras.optimizers.Adam(0.00005)
dec_opt = keras.optimizers.Adam(0.00005)

def train_step(images, secrets):
    with tf.GradientTape(persistent=True) as tape:
        residual = encoder([secrets, images], training=True)
        encoded = images + residual
        decoded = decoder(encoded, training=True)

        secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
        l2_loss = tf.reduce_mean(tf.square(residual))
        total_loss = 1.5 * secret_loss + 0.0 * l2_loss

    enc_grads = tape.gradient(total_loss, encoder.trainable_variables)
    dec_grads = tape.gradient(total_loss, decoder.trainable_variables)

    enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))

    return secret_loss, l2_loss, total_loss, residual

print()
print("="*70)
print(f"Running {TEST_STEPS} test steps...")
print("="*70)
print()

loss_history = []
residual_history = []
issues = []

for step in range(TEST_STEPS):
    images, secrets = get_img_batch(BATCH_SIZE)
    images = tf.constant(images)
    secrets = tf.constant(secrets)

    sec_loss, l2_loss, tot_loss, residual = train_step(images, secrets)

    sec_loss_val = float(sec_loss.numpy())
    res_mean = float(tf.reduce_mean(tf.abs(residual)).numpy())

    loss_history.append(sec_loss_val)
    residual_history.append(res_mean)

    if (step + 1) % 50 == 0:
        print(f"Step {step+1:3d}/{TEST_STEPS}: Secret loss={sec_loss_val:.6f} | Residual mag={res_mean:.6f}")

print()
print("="*70)
print("ANALYSIS")
print("="*70)
print()

loss_start = np.mean(loss_history[0:10])
loss_end = np.mean(loss_history[-10:])
loss_improvement = loss_start - loss_end

print(f"Check 1: Loss Directionality")
print(f"  Initial loss (steps 0-10):  {loss_start:.6f}")
print(f"  Final loss (steps {TEST_STEPS-10}-{TEST_STEPS}):  {loss_end:.6f}")
print(f"  Improvement:              {loss_improvement:+.6f}")

if loss_improvement > 0.001:
    print(f"  ✓ PASS: Loss is DECREASING (improvement = {loss_improvement:.6f})")
elif loss_improvement > -0.001:
    print(f"  ⚠ MARGINAL: Loss is roughly stable (change = {loss_improvement:+.6f})")
    issues.append("Loss not decreasing significantly")
else:
    print(f"  ✗ FAIL: Loss is INCREASING (got worse by {-loss_improvement:.6f})")
    issues.append("Loss is increasing - training not working!")

print()

res_start = np.mean(residual_history[0:10])
res_end = np.mean(residual_history[-10:])
res_growth = res_end - res_start

print(f"Check 2: Residual Magnitude")
print(f"  Initial residual mag (steps 0-10):  {res_start:.6f}")
print(f"  Final residual mag (steps {TEST_STEPS-10}-{TEST_STEPS}):  {res_end:.6f}")
print(f"  Growth:                           {res_growth:+.6f}")

if res_end > 0.01:
    print(f"  ✓ PASS: Residuals are substantial ({res_end:.6f})")
else:
    print(f"  ✗ FAIL: Residuals too small ({res_end:.6f})")
    issues.append("Residuals collapsed to near-zero")

print()

if loss_end < 0.69:
    print(f"Check 3: Random Baseline")
    print(f"  Current loss:     {loss_end:.6f}")
    print(f"  Random baseline:  0.6931")
    print(f"  ✓ PASS: Below random baseline (learning is happening)")
else:
    print(f"Check 3: Random Baseline")
    print(f"  Current loss:     {loss_end:.6f}")
    print(f"  Random baseline:  0.6931")
    print(f"  ✗ FAIL: At/above random baseline (not learning!)")
    issues.append("Loss at random baseline - model not learning")

print()
print("="*70)
print("RESULT")
print("="*70)
print()

if len(issues) == 0:
    print("✓ ALL CHECKS PASSED!")
    print()
    print("256-bit training directionality is CONFIRMED WORKING:")
    print(f"  • Secret loss decreasing: {loss_start:.6f} → {loss_end:.6f}")
    print(f"  • Residuals substantial: {res_end:.6f}")
    print(f"  • Below random baseline: {loss_end:.6f} < 0.6931")
    print()
    print("Safe to proceed with full 280k-step 256-bit training.")
    exit(0)
else:
    print("✗ ISSUES DETECTED:")
    for issue in issues:
        print(f"  • {issue}")
    print()
    print("DO NOT proceed with full training until issues are resolved.")
    exit(1)
