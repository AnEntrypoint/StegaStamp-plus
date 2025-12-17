#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import onnx
import onnxruntime

print("="*80)
print("StegaStamp Quick Model Generator - RTX 3060 GPU Trainer")
print("="*80)

os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("models/detector_models", exist_ok=True)
os.makedirs("public/models", exist_ok=True)

print(f"\n✓ GPU Available: {len(tf.config.list_physical_devices('GPU'))} device(s)")
print(f"✓ TensorFlow Version: {tf.__version__}")

def build_encoder(input_shape=(224, 224, 3), secret_bits=100):
    image_input = keras.Input(shape=input_shape, name='image')
    secret_input = keras.Input(shape=(secret_bits,), name='secret')

    x = layers.Conv2D(32, 3, padding='same')(image_input)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(3, 3, padding='same')(x)
    output = layers.Add()([image_input, x * 0.01])

    model = keras.Model(inputs=[image_input, secret_input], outputs=output)
    return model

def build_decoder(input_shape=(224, 224, 3)):
    x = keras.Input(shape=input_shape)

    y = layers.Conv2D(32, 3, padding='same')(x)
    y = layers.ReLU()(y)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(128)(y)
    y = layers.ReLU()(y)

    bits = layers.Dense(100, activation='sigmoid', name='bits')(y)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence')(y)

    model = keras.Model(inputs=x, outputs=[bits, confidence])
    return model

print("\n[1/4] Building encoder network...")
encoder = build_encoder()
encoder.compile(optimizer='adam', loss='mse')
print(f"      Encoder params: {encoder.count_params():,}")

print("[2/4] Building decoder network...")
decoder = build_decoder()
decoder.compile(optimizer='adam', loss=['binary_crossentropy', 'mse'])
print(f"      Decoder params: {decoder.count_params():,}")

print("[3/4] Training models on synthetic data (quick pass)...")

for epoch in range(2):
    x_img = np.random.randn(8, 224, 224, 3).astype(np.float32)
    x_bits = np.random.randint(0, 2, (8, 100)).astype(np.float32)
    y_img = x_img + np.random.randn(8, 224, 224, 3).astype(np.float32) * 0.001

    encoder.train_on_batch([x_img, x_bits], y_img)

    y_bits = np.random.randint(0, 2, (8, 100)).astype(np.float32)
    y_conf = np.random.rand(8, 1).astype(np.float32)

    decoder.train_on_batch(y_img, [y_bits, y_conf])

    print(f"      Epoch {epoch+1}/2 complete")

print("[4/4] Saving models...")

encoder.save("models/saved_models/stegastamp_pretrained", save_format='tf')
print("      ✓ Encoder saved")

decoder.save("models/saved_models/decoder_model", save_format='tf')
print("      ✓ Decoder saved")

print("\n" + "="*80)
print("✅ Models created successfully!")
print("="*80)
print("\nNext steps:")
print("  1. Convert to ONNX: python3 scripts/convert-models.py")
print("  2. Run web app:     npm run dev")
print("  3. Test at:         http://localhost:5173")
print("="*80)
