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
        residual = self.residual(conv9)
        
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

@tf.function
def train_step(images, secrets, step):
    with tf.GradientTape(persistent=True) as tape:
        residual = encoder([secrets, images], training=True)
        encoded = images + residual
        decoded = decoder(encoded, training=True)
        
        secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
        l2_loss = tf.reduce_mean(tf.square(residual))
        
        total_loss = secret_loss + 1.5 * l2_loss
    
    enc_grads = tape.gradient(total_loss, encoder.trainable_variables)
    dec_grads = tape.gradient(total_loss, decoder.trainable_variables)
    
    enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))
    
    return secret_loss, l2_loss, total_loss

print(f"Training 100-bit secrets for {NUM_STEPS} steps...", flush=True)
for step in range(NUM_STEPS):
    images, secrets = get_img_batch(BATCH_SIZE)
    images = tf.constant(images)
    secrets = tf.constant(secrets)
    
    sec_loss, l2_loss, tot_loss = train_step(images, secrets, step)
    
    if (step + 1) % 500 == 0:
        print(f"Step {step+1}/{NUM_STEPS} Secret:{float(sec_loss):.4f} L2:{float(l2_loss):.4f} Total:{float(tot_loss):.4f}", flush=True)
    
    if (step + 1) % 10000 == 0:
        encoder.save(f'encoder_100bit_step_{step+1}.keras')
        decoder.save(f'decoder_100bit_step_{step+1}.keras')
        print(f"✓ Checkpoint saved at step {step+1}", flush=True)

encoder.save('encoder_100bit_final.keras')
decoder.save('decoder_100bit_final.keras')
print("✓ Training complete!", flush=True)
