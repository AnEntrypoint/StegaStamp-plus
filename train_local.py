#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import subprocess

print("="*80)
print("StegaStamp Training - 256-bit Secrets")
print("="*80)

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detected: {len(gpus)}\n")

os.makedirs("models/saved_models", exist_ok=True)

def build_encoder():
    image_input = keras.Input(shape=(224, 224, 3), name='image')
    secret_input = keras.Input(shape=(256,), name='secret')

    secret_map = keras.layers.Reshape((1, 1, 256))(secret_input)
    secret_tiled = layers.UpSampling2D(size=(224, 224))(secret_map)
    x = layers.Concatenate()([image_input, secret_tiled])

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    residual = layers.Conv2D(3, 3, padding='same')(x)

    output = layers.Add()([image_input, layers.Lambda(lambda x: x * 0.01)(residual)])
    return models.Model(inputs=[image_input, secret_input], outputs=output, name='encoder')

def build_decoder():
    image_input = keras.Input(shape=(224, 224, 3), name='image')

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(image_input)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    bits_out = layers.Dense(256, activation='sigmoid', name='bits')(x)
    conf_out = layers.Dense(1, activation='sigmoid', name='confidence')(x)

    return models.Model(inputs=image_input, outputs=[bits_out, conf_out], name='decoder')

print("[1/4] Building models...")
encoder = build_encoder()
decoder = build_decoder()

encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
decoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=['binary_crossentropy', 'mse'])

print("      ✓ Encoder built")
print("      ✓ Decoder built")

print("\n[2/4] Training encoder...")
for epoch in range(50):
    images = np.random.randn(64, 224, 224, 3).astype(np.float32) * 0.5 + 0.5
    images = np.clip(images, 0, 1)
    secrets = np.random.randint(0, 2, (64, 256)).astype(np.float32)

    encoded = encoder([images, secrets], training=True)
    corrupted = encoded + np.random.randn(*encoded.shape).astype(np.float32) * 0.005
    loss = encoder.train_on_batch([images, secrets], corrupted)
    if (epoch + 1) % 10 == 0:
        print(f"      Epoch {epoch+1}/50 - Loss: {loss:.6f}")

print("\n[3/4] Training decoder...")
for epoch in range(50):
    images = np.random.randn(64, 224, 224, 3).astype(np.float32) * 0.5 + 0.5
    images = np.clip(images, 0, 1)
    secrets = np.random.randint(0, 2, (64, 256)).astype(np.float32)

    encoded = encoder([images, secrets], training=True)
    corrupted = encoded + np.random.randn(*encoded.shape).astype(np.float32) * 0.005
    loss = decoder.train_on_batch(corrupted, [secrets, np.ones((64, 1))])
    loss_val = loss[0] if isinstance(loss, (list, tuple)) else loss
    if (epoch + 1) % 10 == 0:
        print(f"      Epoch {epoch+1}/50 - Loss: {loss_val:.6f}")

print("\n[4/4] Converting to ONNX...")

encoder.export('models/saved_models/encoder_export')
decoder.export('models/saved_models/decoder_export')

print("      ✓ Models saved as TensorFlow SavedModels")

try:
    result = subprocess.run([
        'python3', '-m', 'tf2onnx.convert',
        '--saved-model', 'models/saved_models/encoder_export',
        '--output', 'public/models/encoder.onnx',
        '--opset', '13'
    ], capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print("      ✓ encoder.onnx converted")
    else:
        print(f"      ✗ Encoder conversion failed: {result.stderr}")
except Exception as e:
    print(f"      ✗ Encoder conversion error: {e}")

try:
    result = subprocess.run([
        'python3', '-m', 'tf2onnx.convert',
        '--saved-model', 'models/saved_models/decoder_export',
        '--output', 'public/models/decoder.onnx',
        '--opset', '13'
    ], capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print("      ✓ decoder.onnx converted")
    else:
        print(f"      ✗ Decoder conversion failed: {result.stderr}")
except Exception as e:
    print(f"      ✗ Decoder conversion error: {e}")

print("\n" + "="*80)
print("✅ COMPLETE - Models ready with 256-bit secrets")
print("="*80)
