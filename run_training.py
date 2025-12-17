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

print("Building models...")
encoder = keras.Sequential([
    keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')
])

decoder = keras.Sequential([
    keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(SECRET_BITS, activation='sigmoid')
])

print(f"Encoder params: {encoder.count_params():,}")
print(f"Decoder params: {decoder.count_params():,}")

start_time = time.time()

print(f"\n=== PHASE 1: Train Encoder ({ENC_EPOCHS} epochs) ===")
encoder.compile(optimizer=keras.optimizers.Adam(0.0001), loss='mse')
encoder.fit(train_images, train_images, epochs=ENC_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
print(f"✓ Encoder trained")

print(f"\n=== PHASE 2: Train Decoder ({DEC_EPOCHS} epochs) ===")
decoder.compile(optimizer=keras.optimizers.Adam(0.0001), loss='binary_crossentropy')
encoded_images = encoder.predict(train_images, verbose=0)
for dec_epoch in range(DEC_EPOCHS):
    idx = np.random.permutation(len(encoded_images))
    epoch_start = time.time()
    for i in range(0, len(idx), BATCH_SIZE):
        batch_idx = idx[i:i+BATCH_SIZE]
        batch_enc = encoded_images[batch_idx]
        batch_secs = train_secrets[batch_idx]
        decoder.train_on_batch(batch_enc, batch_secs)
    epoch_time = time.time() - epoch_start
    if (dec_epoch + 1) % 10 == 0:
        print(f"Decoder Epoch {dec_epoch+1}/{DEC_EPOCHS} - {epoch_time:.1f}s")

print("\nTesting on validation set...")
test_images = np.random.rand(500, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
test_secrets = (np.random.rand(500, SECRET_BITS) > 0.5).astype(np.float32)

test_encoded = encoder.predict(test_images, verbose=0)
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
