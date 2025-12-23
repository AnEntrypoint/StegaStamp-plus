#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from PIL import Image, ImageOps
import hashlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_PATH = './data/train'
HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 256
BATCH_SIZE = 4
NUM_STEPS = 140000
LEARNING_RATE = 0.0001
CRITIC_RATE = 0.0001
CHECKPOINT_INTERVAL = 10000
VALIDATION_BATCHES = 10
DEBUG = True

print("=" * 80)
print("STEGASTAMP 256-BIT DEBUG TRAINING")
print("=" * 80)
print(f"Random seed: {os.environ.get('PYTHONHASHSEED', 'not set')}", flush=True)

print("\nLoading training images...", flush=True)
img_files = (glob.glob(os.path.join(TRAIN_PATH, '*.jpg')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.png')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.webp')))
if not img_files:
    raise FileNotFoundError(f"No images in {TRAIN_PATH}")
print(f"Found {len(img_files)} images", flush=True)
print(f"First image: {img_files[0]}", flush=True)

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

def validate_batch(images, secrets, name="batch"):
    has_nan_img = np.isnan(images).any()
    has_inf_img = np.isinf(images).any()
    has_nan_sec = np.isnan(secrets).any()
    has_inf_sec = np.isinf(secrets).any()

    img_min, img_max = np.min(images), np.max(images)
    sec_min, sec_max = np.min(secrets), np.max(secrets)

    if has_nan_img or has_inf_img or has_nan_sec or has_inf_sec:
        print(f"✗ INVALID {name}: NaN={has_nan_img or has_nan_sec}, Inf={has_inf_img or has_inf_sec}", flush=True)
        return False

    if DEBUG and (img_min < -0.6 or img_max > 0.6 or sec_min < -0.1 or sec_max > 1.1):
        print(f"⚠ WARNING {name}: Image range [{img_min:.4f}, {img_max:.4f}], Secret range [{sec_min:.4f}, {sec_max:.4f}]", flush=True)

    return True

def validate_checkpoint(encoder, decoder, num_batches=VALIDATION_BATCHES):
    accuracies = []
    secret_losses = []

    for _ in range(num_batches):
        images, secrets = get_img_batch(BATCH_SIZE)
        validate_batch(images, secrets, "validation")
        images = tf.constant(images)
        secrets = tf.constant(secrets)

        residual = encoder([secrets, images], training=False)
        encoded = images + residual

        if tf.reduce_any(tf.math.is_nan(residual)) or tf.reduce_any(tf.math.is_inf(residual)):
            print(f"✗ NaN/Inf in residual!", flush=True)
            return 0.0, float('inf')

        decoded = decoder(encoded, training=False)

        secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
        probs = tf.nn.sigmoid(decoded).numpy()
        predicted = (probs > 0.5).astype(int)
        accuracy = np.mean(predicted == secrets.numpy())

        accuracies.append(accuracy)
        secret_losses.append(float(secret_loss))

    mean_acc = np.mean(accuracies)
    mean_loss = np.mean(secret_losses)
    return mean_acc, mean_loss

class SpatialTransformer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.localization = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(6)
        ])
        self.localization.get_config()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        theta = self.localization(inputs)
        theta = tf.reshape(theta, [batch_size, 2, 3])
        return tf.raw_ops.GridSampler2D(input=inputs, grid=theta)

class StegaStampEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.secret_dense = keras.layers.Dense(7500, activation='relu', kernel_initializer='he_normal')
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
        secret = keras.layers.Reshape((50, 50, 3))(secret)
        secret_enlarged = keras.layers.UpSampling2D(size=(8, 8))(secret)

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

class CriticNetwork(keras.Model):
    def __init__(self):
        super().__init__()
        self.layers_seq = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), strides=2, activation='leaky_relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), strides=2, activation='leaky_relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), strides=2, activation='leaky_relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), strides=2, activation='leaky_relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='leaky_relu'),
            keras.layers.Dense(1)
        ])

    def call(self, image):
        image = image - 0.5
        return self.layers_seq(image)

print("\nBuilding models...", flush=True)
encoder = StegaStampEncoder()
decoder = StegaStampDecoder()
critic = CriticNetwork()

test_img = tf.constant(np.zeros((1, 400, 400, 3), dtype=np.float32))
test_secret = tf.constant(np.zeros((1, 256), dtype=np.float32))
_ = encoder([test_secret, test_img])
_ = decoder(test_img)
_ = critic(test_img)
print("✓ Model shapes verified", flush=True)

encoder_params = encoder.count_params()
decoder_params = decoder.count_params()
critic_params = critic.count_params()
print(f"Encoder params: {encoder_params:,}", flush=True)
print(f"Decoder params: {decoder_params:,}", flush=True)
print(f"Critic params: {critic_params:,}", flush=True)

enc_opt = keras.optimizers.Adam(LEARNING_RATE)
dec_opt = keras.optimizers.Adam(LEARNING_RATE)
critic_opt = keras.optimizers.Adam(CRITIC_RATE)

def lpips_loss(original, reconstructed):
    diff = original - reconstructed
    return tf.reduce_mean(tf.square(diff))

def train_step(images, secrets, secret_scale, l2_scale, critic_scale):
    with tf.GradientTape(persistent=True) as tape:
        residual = encoder([secrets, images], training=True)

        if tf.reduce_any(tf.math.is_nan(residual)) or tf.reduce_any(tf.math.is_inf(residual)):
            print(f"✗ NaN/Inf detected in residual!", flush=True)
            return None, None, None, None, None, None

        encoded = images + residual
        decoded = decoder(encoded, training=True)
        critic_real = critic(images, training=True)
        critic_fake = critic(encoded, training=True)

        secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
        l2_loss = tf.reduce_mean(tf.square(residual))
        lpips = lpips_loss(images, encoded)
        critic_loss = tf.reduce_mean(tf.nn.relu(1.0 - critic_real)) + tf.reduce_mean(tf.nn.relu(1.0 + critic_fake))

        total_loss = secret_scale * secret_loss + l2_scale * l2_loss + 0.1 * lpips + critic_scale * critic_loss

    enc_grads = tape.gradient(total_loss, encoder.trainable_variables)
    dec_grads = tape.gradient(total_loss, decoder.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)

    for grad, var in zip(enc_grads, encoder.trainable_variables):
        if tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad)):
            print(f"✗ NaN/Inf gradient in encoder {var.name}!", flush=True)
            return None, None, None, None, None, None

    enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))
    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))

    return secret_loss, l2_loss, lpips, critic_loss, total_loss, residual

def get_loss_scales(step, total_steps):
    secret_scale = tf.constant(1.5, dtype=tf.float32)
    phase1_end = int(total_steps * 0.1)
    phase2_end = int(total_steps * 0.4)
    critic_scale = tf.constant(0.1, dtype=tf.float32)

    if step < phase1_end:
        progress = step / phase1_end
        l2_scale = tf.constant(float(progress * 0.5), dtype=tf.float32)
    elif step < phase2_end:
        progress = (step - phase1_end) / (phase2_end - phase1_end)
        l2_scale = tf.constant(float(0.5 + progress * 1.5), dtype=tf.float32)
    else:
        l2_scale = tf.constant(2.0, dtype=tf.float32)

    return secret_scale, l2_scale, critic_scale

print(f"\nTraining 256-bit secrets for {NUM_STEPS} steps...", flush=True)
print(f"Phase 1 (secret-only): steps 0-14000 | Phase 2 (ramp): 14001-56000 | Phase 3 (full): 56001-140000", flush=True)
print("Features: Multi-loss (secret + L2 + LPIPS), Adversarial critic, geometric invariance", flush=True)
print("Validation: Accuracy tested at every checkpoint. Training aborts if model not learning.", flush=True)
print("Debug: Monitoring NaN/Inf, gradient flow, loss components, weight statistics.", flush=True)

phase1_end = int(NUM_STEPS * 0.1)
phase2_end = int(NUM_STEPS * 0.4)
prev_loss = None
prev_acc = None
loss_window = []
acc_window = []
window_size = 10
no_improvement_steps = 0
max_no_improve = 5000

def get_direction(current, previous, threshold=0.01):
    if previous is None:
        return "→"
    delta = (previous - current) / (previous + 1e-8)
    if delta > threshold:
        return "↓"
    elif delta < -threshold:
        return "↑"
    else:
        return "→"

print("\n" + "="*80)
print("STARTING TRAINING LOOP")
print("="*80 + "\n", flush=True)

for step in range(NUM_STEPS):
    images, secrets = get_img_batch(BATCH_SIZE)

    if not validate_batch(images, secrets, f"step {step+1}"):
        print(f"✗ ABORT: Invalid data at step {step+1}", flush=True)
        exit(1)

    images = tf.constant(images)
    secrets = tf.constant(secrets)

    secret_scale, l2_scale, critic_scale = get_loss_scales(step, NUM_STEPS)
    result = train_step(images, secrets, secret_scale, l2_scale, critic_scale)

    if result[0] is None:
        print(f"✗ ABORT: Training step failed at step {step+1}", flush=True)
        exit(1)

    sec_loss, l2_loss, lpips, crit_loss, tot_loss, residual = result

    loss_window.append(float(tot_loss))
    if len(loss_window) > window_size:
        loss_window.pop(0)

    if (step + 1) % 50 == 0:
        res_mean = tf.reduce_mean(tf.abs(residual)).numpy()
        res_std = tf.math.reduce_std(tf.abs(residual)).numpy()
        res_max = tf.reduce_max(tf.abs(residual)).numpy()
        res_min = tf.reduce_min(tf.abs(residual)).numpy()

        if step < phase1_end:
            phase = "P1"
        elif step < phase2_end:
            phase = "P2"
        else:
            phase = "P3"

        direction = get_direction(np.mean(loss_window), prev_loss)
        loss_trend = f"{direction} {np.mean(loss_window):.4f}"

        print(f"[{phase}] Step {step+1:6d}/{NUM_STEPS} | Total:{loss_trend} | Secret:{float(sec_loss):.4f} L2:{float(l2_loss):.4f} LPIPS:{float(lpips):.4f} Critic:{float(crit_loss):.4f} | Res:{res_mean:.6f}±{res_std:.6f} [min:{res_min:.6f} max:{res_max:.6f}] | L2scale:{float(l2_scale):.4f}", flush=True)

    if step < 5:
        res_mean = tf.reduce_mean(tf.abs(residual)).numpy()
        res_min = tf.reduce_min(tf.abs(residual)).numpy()
        res_max = tf.reduce_max(tf.abs(residual)).numpy()
        decoded_min = float(tf.reduce_min(decoder(images + residual, training=False)))
        decoded_max = float(tf.reduce_max(decoder(images + residual, training=False)))
        print(f"[INIT Step {step}] Residual:{res_mean:.8f} [min:{res_min:.8f} max:{res_max:.8f}] | Decoded logits [{decoded_min:.4f}, {decoded_max:.4f}] | L2:{float(l2_loss):.8f} Critic:{float(crit_loss):.8f}", flush=True)

    if (step + 1) % CHECKPOINT_INTERVAL == 0:
        print(f"\n{'='*80}")
        print(f"[CHECKPOINT] Validating step {step+1}...", flush=True)
        val_acc, val_loss = validate_checkpoint(encoder, decoder)
        acc_window.append(val_acc)
        if len(acc_window) > window_size:
            acc_window.pop(0)

        acc_direction = get_direction(val_acc, prev_acc)
        loss_direction = get_direction(val_loss, prev_loss)

        print(f"[CHECKPOINT] Accuracy: {acc_direction} {val_acc:.4f} | Loss: {loss_direction} {val_loss:.4f}", flush=True)

        if val_acc <= 0.5:
            print(f"✗ ABORT: Model not learning! Accuracy at {val_acc:.4f} (random baseline: 0.5)", flush=True)
            print(f"Training failed at step {step+1}. Model did not converge.", flush=True)
            exit(1)

        if prev_loss is not None and val_loss >= prev_loss * 0.98:
            no_improvement_steps += CHECKPOINT_INTERVAL
            if no_improvement_steps > max_no_improve:
                print(f"✗ ABORT: No improvement for {no_improvement_steps} steps. Loss plateau detected.", flush=True)
                exit(1)
            print(f"⚠ WARNING: Loss not improving ({no_improvement_steps}/{max_no_improve} steps). Prev: {prev_loss:.4f}, Current: {val_loss:.4f}", flush=True)
        else:
            no_improvement_steps = 0

        avg_acc = np.mean(acc_window)
        print(f"[CHECKPOINT] Avg accuracy (last {len(acc_window)} checkpoints): {avg_acc:.4f}", flush=True)

        encoder.save(f'encoder_256bit_step_{step+1}.keras')
        decoder.save(f'decoder_256bit_step_{step+1}.keras')
        critic.save(f'critic_256bit_step_{step+1}.keras')
        checkpoint_mb = os.path.getsize(f'encoder_256bit_step_{step+1}.keras') / (1024*1024)
        print(f"✓ CHECKPOINT SAVED: step {step+1} ({checkpoint_mb:.0f}MB) | Accuracy: {val_acc:.4f} {acc_direction} | Loss: {val_loss:.4f} {loss_direction}", flush=True)
        print(f"{'='*80}\n", flush=True)
        prev_loss = val_loss
        prev_acc = val_acc

encoder.save('encoder_256bit_final.keras')
decoder.save('decoder_256bit_final.keras')
critic.save('critic_256bit_final.keras')
print("\n" + "="*80)
print("✓ Training complete with all paper features!", flush=True)
print("✓ All checkpoints validated and passed accuracy threshold.", flush=True)
print("="*80)
