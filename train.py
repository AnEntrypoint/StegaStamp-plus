#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import glob
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LD_LIBRARY_PATH'] = '/home/user/diffusers/pixel_art_venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.6/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.6'

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU: {len(physical_devices)} device(s)")

np.random.seed(42)
tf.random.set_seed(42)

TRAIN_PATH = '/home/user/StegaStamp-plus/data/train'
HEIGHT, WIDTH = 256, 256
SECRET_SIZE = 8
BATCH_SIZE = 2
NUM_STEPS = 2000
LEARNING_RATE = 0.001

print(f"Loading images from {TRAIN_PATH}...")
img_files = (glob.glob(os.path.join(TRAIN_PATH, '*.jpg')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.png')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.webp')))
if not img_files:
    raise FileNotFoundError(f"No images in {TRAIN_PATH}")

cached_images = []
for i, img_path in enumerate(img_files):
    img = keras.preprocessing.image.load_img(img_path, target_size=(HEIGHT, WIDTH))
    img = keras.preprocessing.image.img_to_array(img) / 255.0
    cached_images.append(img.astype(np.float32))

cached_images = np.array(cached_images, dtype=np.float32)
print(f"✓ Cached {len(cached_images)} images ({HEIGHT}x{WIDTH})")

def get_img_batch(batch_size=BATCH_SIZE, step=0):
    indices = np.random.choice(len(cached_images), batch_size, replace=True)
    images = cached_images[indices]

    if step < 100:
        secrets = np.ones((batch_size, SECRET_SIZE), dtype=np.float32) * 0.7
    elif step < 200:
        secrets = (np.random.rand(batch_size, SECRET_SIZE) * 0.2 + 0.6).astype(np.float32)
    else:
        secrets = (np.random.binomial(1, 0.5, (batch_size, SECRET_SIZE))).astype(np.float32)

    return tf.constant(images), tf.constant(secrets)

class Encoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.secret_dense = keras.layers.Dense(64*64*3, activation='relu')
        self.conv1 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.final = keras.layers.Conv2D(3, 1, padding='same')

    def call(self, inputs):
        images, secrets = inputs
        batch_size = tf.shape(images)[0]
        secret_expanded = self.secret_dense(secrets)
        secret_expanded = tf.reshape(secret_expanded, (batch_size, 64, 64, 3))
        secret_expanded = tf.image.resize(secret_expanded, (HEIGHT, WIDTH))
        x = tf.concat([images, secret_expanded], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final(x)
        residual = tf.nn.tanh(x) * 0.1
        return images + residual

class Decoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.conv3 = keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(SECRET_SIZE)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

print(f"Building models (secret_size={SECRET_SIZE}, img={HEIGHT}x{WIDTH})...")
encoder = Encoder()
decoder = Decoder()

enc_opt = keras.optimizers.Adam(LEARNING_RATE)
dec_opt = keras.optimizers.Adam(LEARNING_RATE)

def train_step(images, secrets, step):
    with tf.GradientTape(persistent=True) as tape:
        encoded = encoder([images, secrets], training=True)
        decoded = decoder(encoded, training=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))

    enc_grads = tape.gradient(loss, encoder.trainable_variables)
    dec_grads = tape.gradient(loss, decoder.trainable_variables)
    enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))

    return loss

print(f"Training {NUM_STEPS} steps with curriculum learning (constant→random)...")
start_time = time.time()

for step in range(NUM_STEPS):
    images, secrets = get_img_batch(BATCH_SIZE, step)
    loss = train_step(images, secrets, step)

    if (step + 1) % 100 == 0:
        elapsed = (time.time() - start_time) / 60
        phase = "const" if step < 100 else ("grad" if step < 200 else "random")
        print(f"Step {step+1}/{NUM_STEPS} [{phase:5s}] - Loss: {float(loss):.6f} ({elapsed:.1f}m)")

print("\nTesting...")
test_images, test_secrets = get_img_batch(BATCH_SIZE, NUM_STEPS)
test_encoded = encoder([test_images, test_secrets], training=False)
test_decoded = decoder(test_encoded, training=False)
test_pred = (test_decoded.numpy() > 0.5).astype(int)
test_true = (test_secrets.numpy() > 0.5).astype(int)
accuracy = np.mean(test_pred == test_true)
print(f"Test accuracy: {accuracy*100:.2f}%")

encoder.save('encoder.keras')
decoder.save('decoder.keras')
print("✓ Done")
