import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

class StegaStampEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.secret_dense = keras.layers.Dense(7500, activation='relu', kernel_initializer='he_normal')
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
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
        residual = self.residual(conv2)
        return residual

class SimpleDecoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(100)

    def call(self, image):
        image = image - 0.5
        x = self.conv1(image)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

print("Testing encoder output...")
encoder = StegaStampEncoder()
decoder = SimpleDecoder()

test_image = np.random.rand(2, 400, 400, 3).astype(np.float32)
test_secret = np.random.binomial(1, 0.5, (2, 100)).astype(np.float32)

residual = encoder([tf.constant(test_secret), tf.constant(test_image)], training=True)
print(f"Residual shape: {residual.shape}")
print(f"Residual mean: {tf.reduce_mean(residual).numpy():.6f}")
print(f"Residual std: {tf.math.reduce_std(residual).numpy():.6f}")
print(f"Residual max: {tf.reduce_max(tf.abs(residual)).numpy():.6f}")

l2_loss = tf.reduce_mean(tf.square(residual))
print(f"L2 loss: {l2_loss.numpy():.6f}")

encoded = test_image + residual
decoded = decoder(encoded, training=True)
print(f"Decoded shape: {decoded.shape}")
print(f"Decoded mean: {tf.reduce_mean(decoded).numpy():.6f}")
print(f"Decoded std: {tf.math.reduce_std(decoded).numpy():.6f}")

secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_secret, logits=decoded))
print(f"Secret loss: {secret_loss.numpy():.6f}")

print("\nTesting gradient flow...")
with tf.GradientTape() as tape:
    residual = encoder([tf.constant(test_secret), tf.constant(test_image)], training=True)
    encoded = test_image + residual
    decoded = decoder(encoded, training=True)
    loss = secret_loss + 1.5 * l2_loss

enc_grads = tape.gradient(loss, encoder.trainable_variables)
print(f"Encoder gradient count: {len(enc_grads)}")
print(f"Non-None gradients: {sum(1 for g in enc_grads if g is not None)}")
if enc_grads[0] is not None:
    print(f"First gradient norm: {tf.norm(enc_grads[0]).numpy():.6f}")
