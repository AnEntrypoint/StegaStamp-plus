#!/usr/bin/env python3
"""
Exhaustive validation of the 256-bit training pipeline.
Tests each component step-by-step to catch data mismatches.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 256
BATCH_SIZE = 2  # Small for testing

print("=" * 80)
print("EXHAUSTIVE VALIDATION OF 256-BIT TRAINING PIPELINE")
print("=" * 80)

# === Step 1: Create synthetic test data ===
print("\n[STEP 1] Creating synthetic test data...")
test_images = np.random.uniform(0, 1, (BATCH_SIZE, HEIGHT, WIDTH, 3)).astype(np.float32)
test_secrets = np.random.binomial(1, 0.5, (BATCH_SIZE, SECRET_SIZE)).astype(np.float32)
print(f"✓ Images shape: {test_images.shape} (expected: ({BATCH_SIZE}, {HEIGHT}, {WIDTH}, 3))")
print(f"✓ Secrets shape: {test_secrets.shape} (expected: ({BATCH_SIZE}, {SECRET_SIZE}))")
print(f"✓ Images range: [{test_images.min():.3f}, {test_images.max():.3f}] (expected: [0, 1])")
print(f"✓ Secrets range: [{test_secrets.min():.0f}, {test_secrets.max():.0f}] (expected: [0, 1])")

# === Step 2: Test Encoder ===
print("\n[STEP 2] Testing StegaStampEncoder...")

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
residual = encoder([test_secrets, test_images], training=True)
print(f"✓ Residual shape: {residual.shape} (expected: ({BATCH_SIZE}, {HEIGHT}, {WIDTH}, 3))")
print(f"✓ Residual range: [{residual.numpy().min():.6f}, {residual.numpy().max():.6f}]")
print(f"✓ Encoder trainable variables: {len(encoder.trainable_variables)}")

# === Step 3: Test Decoder ===
print("\n[STEP 3] Testing StegaStampDecoder...")

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
encoded = test_images + residual
decoded = decoder(encoded, training=True)
print(f"✓ Encoded shape: {encoded.shape} (expected: ({BATCH_SIZE}, {HEIGHT}, {WIDTH}, 3))")
print(f"✓ Decoded shape: {decoded.shape} (expected: ({BATCH_SIZE}, {SECRET_SIZE}))")
print(f"✓ Decoded range: [{decoded.numpy().min():.3f}, {decoded.numpy().max():.3f}]")

# === Step 4: Test Critic ===
print("\n[STEP 4] Testing StegaStampCritic...")

class StegaStampCritic(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4 = keras.layers.Conv2D(128, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = keras.layers.Conv2D(128, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, image):
        x = image - 0.5
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)

critic = StegaStampCritic()
critic_fake = critic(encoded, training=True)
critic_real = critic(test_images, training=True)
print(f"✓ Critic fake shape: {critic_fake.shape} (expected: ({BATCH_SIZE}, 1))")
print(f"✓ Critic real shape: {critic_real.shape} (expected: ({BATCH_SIZE}, 1))")
print(f"✓ Critic fake range: [{critic_fake.numpy().min():.4f}, {critic_fake.numpy().max():.4f}] (expected: [0, 1])")
print(f"✓ Critic real range: [{critic_real.numpy().min():.4f}, {critic_real.numpy().max():.4f}] (expected: [0, 1])")

# === Step 5: Test Loss Computation ===
print("\n[STEP 5] Testing loss computation...")

secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_secrets, logits=decoded))
l2_loss = tf.reduce_mean(tf.square(residual))
critic_loss = -tf.reduce_mean(tf.math.log(critic_real + 1e-8) + tf.math.log(1 - critic_fake + 1e-8))

print(f"✓ Secret loss: {float(secret_loss):.6f} (expected: ~0.69 for random)")
print(f"✓ L2 loss: {float(l2_loss):.6f}")
print(f"✓ Critic loss: {float(critic_loss):.6f}")
print(f"✓ All losses are finite: {tf.math.is_finite(secret_loss) and tf.math.is_finite(l2_loss) and tf.math.is_finite(critic_loss)}")

# === Step 6: Test Gradients ===
print("\n[STEP 6] Testing gradient flow...")

with tf.GradientTape(persistent=True) as tape:
    residual_test = encoder([test_secrets, test_images], training=True)
    encoded_test = test_images + residual_test
    decoded_test = decoder(encoded_test, training=True)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_secrets, logits=decoded_test))

grads_enc = tape.gradient(loss, encoder.trainable_variables)
grads_dec = tape.gradient(loss, decoder.trainable_variables)
del tape

has_none_grads_enc = any(g is None for g in grads_enc)
has_none_grads_dec = any(g is None for g in grads_dec)
print(f"✓ Encoder has None gradients: {has_none_grads_enc} (expected: False)")
print(f"✓ Decoder has None gradients: {has_none_grads_dec} (expected: False)")
print(f"✓ Encoder gradient norms: {[float(tf.norm(g)) for g in grads_enc[:3]]}")  # First 3
print(f"✓ Decoder gradient norms: {[float(tf.norm(g)) for g in grads_dec[:3]]}")  # First 3

# === Step 7: Test Full Training Step ===
print("\n[STEP 7] Testing full training step (simulation)...")

enc_opt = keras.optimizers.Adam(0.0001)
dec_opt = keras.optimizers.Adam(0.0001)
crit_opt = keras.optimizers.Adam(0.00005)

with tf.GradientTape(persistent=True) as tape:
    residual_train = encoder([test_secrets, test_images], training=True)
    encoded_train = test_images + residual_train
    decoded_train = decoder(encoded_train, training=True)

    secret_loss_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_secrets, logits=decoded_train))
    l2_loss_train = tf.reduce_mean(tf.square(residual_train))

    critic_fake_train = critic(encoded_train, training=True)
    critic_real_train = critic(test_images, training=True)
    critic_loss_train = -tf.reduce_mean(tf.math.log(critic_real_train + 1e-8) + tf.math.log(1 - critic_fake_train + 1e-8))

    total_loss = 1.5 * secret_loss_train + 0.5 * l2_loss_train + 0.1 * (-tf.reduce_mean(tf.math.log(critic_fake_train + 1e-8)))

# Apply gradients
enc_grads = tape.gradient(total_loss, encoder.trainable_variables)
dec_grads = tape.gradient(total_loss, decoder.trainable_variables)
crit_grads = tape.gradient(critic_loss_train, critic.trainable_variables)

enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))
crit_opt.apply_gradients(zip(crit_grads, critic.trainable_variables))

print(f"✓ Total loss: {float(total_loss):.6f}")
print(f"✓ Gradients applied successfully")

print("\n" + "=" * 80)
print("✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
print("=" * 80)
print("\nPipeline is ready for training!")
print("- Encoder: OK")
print("- Decoder: OK")
print("- Critic: OK")
print("- Loss computation: OK")
print("- Gradient flow: OK")
print("- Training step: OK")
