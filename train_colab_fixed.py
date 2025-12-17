import numpy as np
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(physical_devices)}")

SECRET_BITS = 256
IMG_SIZE = 128

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

print("Generating training data...")
train_images = np.random.rand(50, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
train_secrets = (np.random.rand(50, SECRET_BITS) > 0.5).astype(np.float32)

encoder.compile(optimizer='adam', loss='mse')
decoder.compile(optimizer='adam', loss='binary_crossentropy')

print("Training encoder...")
encoder.fit(train_images, train_images, epochs=2, batch_size=16, verbose=1)

print("Training decoder...")
for epoch in range(2):
    encoded = encoder.predict(train_images, verbose=0)
    decoder.fit(encoded, train_secrets, epochs=1, batch_size=16, verbose=1)

test_images = np.random.rand(10, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) * 0.5 + 0.25
test_secrets = (np.random.rand(10, SECRET_BITS) > 0.5).astype(np.float32)

test_encoded = encoder.predict(test_images, verbose=0)
pred_bits = decoder.predict(test_encoded, verbose=0)

pred_binary = (pred_bits > 0.5).astype(int)
test_binary = (test_secrets > 0.5).astype(int)
accuracy = np.mean(pred_binary == test_binary)

print(f"\nTest accuracy: {accuracy*100:.1f}%")
print(f"Bit error rate: {(1-accuracy)*100:.1f}%")

print("\nConverting to ONNX...")
try:
    import subprocess
    subprocess.run(['pip', 'install', 'tf2onnx', '-q'], check=True)
    from tf2onnx import convert

    convert.from_keras(encoder, output_path='encoder.onnx')
    print("✓ Encoder saved to ONNX")

    convert.from_keras(decoder, output_path='decoder.onnx')
    print("✓ Decoder saved to ONNX")
    print("\n✓ Models ready for deployment!")
except Exception as e:
    print(f"ONNX export error: {e}")
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')
    print("Models saved as H5 instead")
