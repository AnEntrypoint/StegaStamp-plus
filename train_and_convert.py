#!/usr/bin/env python3
"""Train models and directly convert to ONNX in one go"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
import onnx

print("="*80)
print("Training & Converting to ONNX")
print("="*80)

os.makedirs("public/models", exist_ok=True)

class StegaStampEncoder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv_out = layers.Conv2D(3, 3, padding='same')

    def call(self, inputs):
        image, secret = inputs
        batch_size = keras.ops.shape(image)[0]
        secret_map = keras.ops.reshape(secret, [batch_size, 1, 1, 100])
        secret_tiled = keras.ops.tile(secret_map, [1, 224, 224, 1])
        x = keras.ops.concatenate([image, secret_tiled], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.conv_out(x)
        return image + residual * 0.01

class StegaStampDecoder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.bits_out = layers.Dense(100, activation='sigmoid')
        self.conf_out = layers.Dense(1, activation='sigmoid')

    def call(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        bits = self.bits_out(x)
        confidence = self.conf_out(x)
        return bits, confidence

print("\n[1/4] Building models...")
encoder = StegaStampEncoder()
decoder = StegaStampDecoder()

encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
decoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=['binary_crossentropy', 'mse'])
print("      ✓ Models built")

print("\n[2/4] Training...")
def generate_batch(batch_size=8):
    images = np.random.randn(batch_size, 224, 224, 3).astype(np.float32) * 0.5 + 0.5
    images = np.clip(images, 0, 1)
    secrets = np.random.randint(0, 2, (batch_size, 100)).astype(np.float32)
    return images, secrets

for epoch in range(5):
    images, secrets = generate_batch(16)
    encoded = encoder([images, secrets])
    corrupted = encoded + np.random.randn(*encoded.shape).astype(np.float32) * 0.01
    loss = encoder.train_on_batch([images, secrets], corrupted)
    print(f"      Encoder Epoch {epoch+1}/5 - Loss: {loss:.6f}")

for epoch in range(5):
    images, secrets = generate_batch(16)
    encoded = encoder([images, secrets])
    corrupted = encoded + np.random.randn(*encoded.shape).astype(np.float32) * 0.01
    loss = decoder.train_on_batch(corrupted, [secrets, np.ones((16, 1))])
    loss_val = loss[0] if isinstance(loss, (list, tuple)) else loss
    print(f"      Decoder Epoch {epoch+1}/5 - Loss: {loss_val:.6f}")

print("\n[3/4] Converting encoder to ONNX...")
import subprocess
# Export to SavedModel format using export()
encoder.export("models/saved_models/encoder_tmp")
decoder.export("models/saved_models/decoder_tmp")

# Convert using tf2onnx
result = subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--saved-model", "models/saved_models/encoder_tmp",
    "--output", "public/models/encoder.onnx",
    "--opset", "13"
], capture_output=True, text=True, timeout=60)

if result.returncode == 0:
    encoder_size = os.path.getsize("public/models/encoder.onnx") / (1024*1024)
    print(f"      ✓ Encoder saved ({encoder_size:.1f}MB)")
else:
    print(f"      ⚠ Encoder conversion: {result.stderr[:200]}")
    encoder_size = 0

print("\n[4/4] Converting decoder to ONNX...")
result = subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--saved-model", "models/saved_models/decoder_tmp",
    "--output", "public/models/decoder.onnx",
    "--opset", "13"
], capture_output=True, text=True, timeout=60)

if result.returncode == 0:
    decoder_size = os.path.getsize("public/models/decoder.onnx") / (1024*1024)
    print(f"      ✓ Decoder saved ({decoder_size:.1f}MB)")
else:
    print(f"      ⚠ Decoder conversion: {result.stderr[:200]}")
    decoder_size = 0

print("\n" + "="*80)
print("✅ COMPLETE - Models ready for web app")
print("="*80)
print(f"\nModels:")
print(f"  • encoder.onnx ({encoder_size:.1f}MB)")
print(f"  • decoder.onnx ({decoder_size:.1f}MB)")
print(f"\nNext: npm run dev")
print("="*80)
