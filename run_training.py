import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

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
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30
BATCH_SIZE = 64
TRAIN_SAMPLES = 500

print(f"Configuration: {TRAIN_SAMPLES} training samples, {PHASE1_EPOCHS + PHASE2_EPOCHS} total epochs")

def build_encoder():
    image_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
    secret_in = keras.Input(shape=(SECRET_BITS,), name='secret')

    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(image_in)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    secret_expanded = keras.layers.Dense(1024, activation='relu')(secret_in)
    secret_expanded = keras.layers.Dense(2048, activation='relu')(secret_expanded)
    secret_expanded = keras.layers.Reshape((16, 16, 8))(secret_expanded)
    secret_expanded = keras.layers.UpSampling2D(size=(8, 8))(secret_expanded)
    secret_expanded = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(secret_expanded)
    secret_expanded = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(secret_expanded)
    secret_expanded = keras.layers.Lambda(lambda x: x * 0.001)(secret_expanded)

    combined = keras.layers.Concatenate()([x, secret_expanded])
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(combined)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    out = keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)

    return keras.Model(inputs=[image_in, secret_in], outputs=out)

def build_decoder():
    image_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')

    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(image_in)
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(512, activation='relu')(x)

    bits_out = keras.layers.Dense(SECRET_BITS, activation='sigmoid', name='bits')(x)

    return keras.Model(inputs=image_in, outputs=bits_out)

print("Building models...")
encoder = build_encoder()
decoder = build_decoder()

print("Generating training data...")
train_images = np.random.rand(TRAIN_SAMPLES, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
train_secrets = (np.random.rand(TRAIN_SAMPLES, SECRET_BITS) > 0.5).astype(np.float32)

print(f"Encoder params: {encoder.count_params():,}")
print(f"Decoder params: {decoder.count_params():,}")

enc_opt = keras.optimizers.Adam(learning_rate=0.0001)
dec_opt = keras.optimizers.Adam(learning_rate=0.0001)

def bit_loss_fn(decoded_bits, secret_bits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=secret_bits,
        logits=tf.math.log(tf.maximum(decoded_bits, 1e-7)) - tf.math.log(tf.maximum(1 - decoded_bits, 1e-7))
    ))

def combined_loss_fn(encoded, original, decoded_bits, secret_bits):
    img_loss = tf.reduce_mean(tf.square(encoded - original))
    secret_loss = bit_loss_fn(decoded_bits, secret_bits)
    return 0.5 * img_loss + 1.0 * secret_loss

import time
total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS
start_time = time.time()

print(f"\n=== PHASE 1: Secret Loss Only ({PHASE1_EPOCHS} epochs) ===")
for epoch in range(PHASE1_EPOCHS):
    idx = np.random.permutation(len(train_images))
    epoch_loss = 0
    num_batches = 0
    epoch_start = time.time()

    for i in range(0, len(idx), BATCH_SIZE):
        batch_idx = idx[i:i+BATCH_SIZE]
        batch_imgs = train_images[batch_idx]
        batch_secs = train_secrets[batch_idx]

        with tf.GradientTape(persistent=True) as tape:
            encoded = encoder([batch_imgs, batch_secs], training=True)
            decoded_bits = decoder(encoded, training=True)
            loss = bit_loss_fn(decoded_bits, batch_secs)

        grads_enc = tape.gradient(loss, encoder.trainable_variables)
        grads_dec = tape.gradient(loss, decoder.trainable_variables)

        enc_opt.apply_gradients(zip(grads_enc, encoder.trainable_variables))
        dec_opt.apply_gradients(zip(grads_dec, decoder.trainable_variables))
        del tape

        epoch_loss += float(loss)
        num_batches += 1

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{total_epochs} - SecretLoss: {epoch_loss/num_batches:.6f} - {epoch_time:.1f}s")

print(f"\n=== PHASE 2: Combined Loss ({PHASE2_EPOCHS} epochs) ===")
for epoch in range(PHASE2_EPOCHS):
    idx = np.random.permutation(len(train_images))
    epoch_loss = 0
    num_batches = 0
    epoch_start = time.time()

    for i in range(0, len(idx), BATCH_SIZE):
        batch_idx = idx[i:i+BATCH_SIZE]
        batch_imgs = train_images[batch_idx]
        batch_secs = train_secrets[batch_idx]

        with tf.GradientTape(persistent=True) as tape:
            encoded = encoder([batch_imgs, batch_secs], training=True)
            decoded_bits = decoder(encoded, training=True)
            loss = combined_loss_fn(encoded, batch_imgs, decoded_bits, batch_secs)

        grads_enc = tape.gradient(loss, encoder.trainable_variables)
        grads_dec = tape.gradient(loss, decoder.trainable_variables)

        enc_opt.apply_gradients(zip(grads_enc, encoder.trainable_variables))
        dec_opt.apply_gradients(zip(grads_dec, decoder.trainable_variables))
        del tape

        epoch_loss += float(loss)
        num_batches += 1

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+PHASE1_EPOCHS+1}/{total_epochs} - Combined: {epoch_loss/num_batches:.6f} - {epoch_time:.1f}s")

print("\nTesting on validation set...")
test_images = np.random.rand(100, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
test_secrets = (np.random.rand(100, SECRET_BITS) > 0.5).astype(np.float32)

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
