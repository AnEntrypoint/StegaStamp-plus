#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from PIL import Image, ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_PATH = './data/train'
HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 100
BATCH_SIZE = 4
NUM_STEPS = 140000
LEARNING_RATE = 0.0001

print("Loading training images...", flush=True)
img_files = (glob.glob(os.path.join(TRAIN_PATH, '*.jpg')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.png')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.webp')))
if not img_files:
    raise FileNotFoundError(f"No images in {TRAIN_PATH}")
print(f"Found {len(img_files)} images", flush=True)

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

print("Building models...", flush=True)
encoder = StegaStampEncoder()
decoder = StegaStampDecoder()

enc_opt = keras.optimizers.Adam(LEARNING_RATE)
dec_opt = keras.optimizers.Adam(LEARNING_RATE)

def train_step(images, secrets, secret_scale, l2_scale):
    with tf.GradientTape(persistent=True) as tape:
        residual = encoder([secrets, images], training=True)
        encoded = images + residual
        decoded = decoder(encoded, training=True)

        secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
        l2_loss = tf.reduce_mean(tf.square(residual))

        total_loss = secret_scale * secret_loss + l2_scale * l2_loss

    enc_grads = tape.gradient(total_loss, encoder.trainable_variables)
    dec_grads = tape.gradient(total_loss, decoder.trainable_variables)

    enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))

    return secret_loss, l2_loss, total_loss, residual

def get_loss_scales(step, total_steps):
    secret_scale = tf.constant(1.5, dtype=tf.float32)
    phase1_end = int(total_steps * 0.1)
    phase2_end = int(total_steps * 0.4)

    if step < phase1_end:
        progress = step / phase1_end
        l2_scale = tf.constant(float(progress * 0.5), dtype=tf.float32)
    elif step < phase2_end:
        progress = (step - phase1_end) / (phase2_end - phase1_end)
        l2_scale = tf.constant(float(0.5 + progress * 1.5), dtype=tf.float32)
    else:
        l2_scale = tf.constant(2.0, dtype=tf.float32)

    return secret_scale, l2_scale

print(f"Training 100-bit secrets for {NUM_STEPS} steps...", flush=True)
print(f"Phase 1 (secret-only): steps 0-14000 | Phase 2 (ramp): 14001-56000 | Phase 3 (full): 56001-140000", flush=True)

phase1_end = int(NUM_STEPS * 0.1)
phase2_end = int(NUM_STEPS * 0.4)

for step in range(NUM_STEPS):
    images, secrets = get_img_batch(BATCH_SIZE)
    images = tf.constant(images)
    secrets = tf.constant(secrets)

    secret_scale, l2_scale = get_loss_scales(step, NUM_STEPS)
    sec_loss, l2_loss, tot_loss, residual = train_step(images, secrets, secret_scale, l2_scale)

    if (step + 1) % 50 == 0:
        res_mean = tf.reduce_mean(tf.abs(residual)).numpy()
        res_max = tf.reduce_max(tf.abs(residual)).numpy()
        if step < phase1_end:
            phase = "P1"
        elif step < phase2_end:
            phase = "P2"
        else:
            phase = "P3"
        print(f"[{phase}] Step {step+1:6d}/{NUM_STEPS} | Secret:{float(sec_loss):.4f} L2:{float(l2_loss):.4f} Total:{float(tot_loss):.4f} | ResidualMag:{res_mean:.6f} ResidualMax:{res_max:.6f} | L2scale:{float(l2_scale):.4f}", flush=True)

    if step < 5:
        res_mean = tf.reduce_mean(tf.abs(residual)).numpy()
        print(f"[INIT Step {step}] Residual:{res_mean:.8f} L2:{float(l2_loss):.8f}", flush=True)

    if (step + 1) % 10000 == 0:
        encoder.save(f'encoder_100bit_step_{step+1}.keras')
        decoder.save(f'decoder_100bit_step_{step+1}.keras')
        checkpoint_mb = os.path.getsize(f'encoder_100bit_step_{step+1}.keras') / (1024*1024)
        print(f"✓ CHECKPOINT saved: step {step+1} ({checkpoint_mb:.0f}MB) | Secret:{float(sec_loss):.4f} L2:{float(l2_loss):.4f}", flush=True)

encoder.save('encoder_100bit_final.keras')
decoder.save('decoder_100bit_final.keras')
print("✓ Training complete!", flush=True)
