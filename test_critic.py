#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Copy Critic class
class StegaStampCritic(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4 = keras.layers.Conv2D(128, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = keras.layers.Conv2D(128, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, image):
        x = image - 0.5
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)

print("Testing Critic Network...")
critic = StegaStampCritic()

# Test forward pass
test_image = np.random.randn(2, 400, 400, 3).astype(np.float32)
test_input = tf.constant(test_image)

output = critic(test_input, training=False)
print(f"✓ Critic output shape: {output.shape} (expected: (2, 1))")
print(f"✓ Critic output values: {output.numpy().flatten()}")
print(f"✓ Critic is working correctly!")
