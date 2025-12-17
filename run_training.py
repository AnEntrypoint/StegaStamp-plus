import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"✓ GPU enabled: {len(physical_devices)} device(s)")
else:
    print("⚠ No GPU found, using CPU")

np.random.seed(42)
tf.random.set_seed(42)

SECRET_BITS = 256
IMG_SIZE = 128
ENC_EPOCHS = 30
DEC_EPOCHS = 40
BATCH_SIZE = 32
TRAIN_SAMPLES = 2000

print(f"Configuration: {TRAIN_SAMPLES} random images, {IMG_SIZE}x{IMG_SIZE}, separate training (enc:{ENC_EPOCHS} + dec:{DEC_EPOCHS} epochs)")

print("Generating training data...")
train_images = np.random.rand(TRAIN_SAMPLES, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
train_secrets = (np.random.rand(TRAIN_SAMPLES, SECRET_BITS) > 0.5).astype(np.float32)

def build_encoder():
    image_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
    secret_in = keras.Input(shape=(SECRET_BITS,), name='secret')

    x = keras.layers.Conv2D(48, 3, padding='same', activation='relu')(image_in)
    x = keras.layers.Conv2D(48, 3, padding='same', activation='relu')(x)

    secret_expanded = keras.layers.Dense(512, activation='relu')(secret_in)
    secret_expanded = keras.layers.Dense(1024, activation='relu')(secret_expanded)
    secret_expanded = keras.layers.Reshape((16, 16, 4))(secret_expanded)
    secret_expanded = keras.layers.UpSampling2D(size=(8, 8))(secret_expanded)
    secret_expanded = keras.layers.Conv2D(48, 3, padding='same', activation='relu')(secret_expanded)
    secret_expanded = keras.layers.Lambda(lambda x: x * 0.001)(secret_expanded)

    combined = keras.layers.Concatenate()([x, secret_expanded])
    x = keras.layers.Conv2D(48, 3, padding='same', activation='relu')(combined)
    x = keras.layers.Conv2D(48, 3, padding='same', activation='relu')(x)
    out = keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)

    return keras.Model(inputs=[image_in, secret_in], outputs=out)

def build_decoder():
    image_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')

    x = keras.layers.Conv2D(96, 3, padding='same', activation='relu')(image_in)
    x = keras.layers.Conv2D(96, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu')(x)

    bits_out = keras.layers.Dense(SECRET_BITS, activation='sigmoid', name='bits')(x)

    return keras.Model(inputs=image_in, outputs=bits_out)

print("Building models...")
encoder = build_encoder()
decoder = build_decoder()

print(f"Encoder params: {encoder.count_params():,}")
print(f"Decoder params: {decoder.count_params():,}")

start_time = time.time()

print(f"\n=== PHASE 1: Train Encoder ({ENC_EPOCHS} epochs) ===")
encoder.compile(optimizer=keras.optimizers.Adam(0.0001), loss='mse')
encoder.fit(train_images, train_images, epochs=ENC_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

print(f"=== PHASE 2: Train Decoder ({DEC_EPOCHS} epochs) ===")
decoder.compile(optimizer=keras.optimizers.Adam(0.0001), loss='binary_crossentropy')
for dec_epoch in range(DEC_EPOCHS):
    epoch_start = time.time()
    idx = np.random.permutation(len(train_images))
    for i in range(0, len(idx), BATCH_SIZE):
        batch_idx = idx[i:i+BATCH_SIZE]
        batch_imgs = train_images[batch_idx]
        batch_secs = train_secrets[batch_idx]
        encoded = encoder([batch_imgs, batch_secs], training=False)
        decoder.train_on_batch(encoded, batch_secs)
    epoch_time = time.time() - epoch_start
    if (dec_epoch + 1) % 5 == 0:
        print(f"Decoder Epoch {dec_epoch+1}/{DEC_EPOCHS} - {epoch_time:.1f}s")

print("\nTesting on validation set...")
test_images = np.random.rand(500, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
test_secrets = (np.random.rand(500, SECRET_BITS) > 0.5).astype(np.float32)

test_encoded = encoder.predict([test_images, test_secrets], verbose=0)
pred_bits = decoder.predict(test_encoded, verbose=0)

pred_binary = (pred_bits > 0.5).astype(int)
test_binary = (test_secrets > 0.5).astype(int)
accuracy = np.mean(pred_binary == test_binary)

print(f"Test accuracy: {accuracy*100:.1f}%")
print(f"Bit error rate: {(1-accuracy)*100:.1f}%")

total_time = time.time() - start_time
print(f"Total training time: {total_time/60:.1f} minutes")

print("\nSaving models...")
encoder.save('encoder.h5')
decoder.save('decoder.h5')
print("✓ Models saved as H5")
