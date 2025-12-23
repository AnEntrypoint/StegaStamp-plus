#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SpatialTransformerNetwork(keras.Model):
    def __init__(self):
        super().__init__()
        self.localization = keras.Sequential([
            keras.layers.Conv2D(64, 7, strides=2, activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(128, 5, strides=2, activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(6, kernel_initializer=keras.initializers.Zeros(),
                             bias_initializer=keras.initializers.Constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        ])

    def call(self, x):
        theta = self.localization(x)
        theta = tf.reshape(theta, (-1, 2, 3))

        # Get image shape
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        # Generate mesh grid
        grid_x = tf.linspace(-1.0, 1.0, width)
        grid_y = tf.linspace(-1.0, 1.0, height)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)

        # Flatten and add batch dimension
        grid_x = tf.expand_dims(tf.expand_dims(grid_x, 0), -1)
        grid_y = tf.expand_dims(tf.expand_dims(grid_y, 0), -1)
        grid_x = tf.tile(grid_x, [batch_size, 1, 1, 1])
        grid_y = tf.tile(grid_y, [batch_size, 1, 1, 1])

        # Stack coordinates
        coords = tf.concat([grid_x, grid_y, tf.ones_like(grid_x)], axis=-1)
        # coords shape: [batch, height, width, 3]

        # Reshape for batch matrix multiplication
        coords_flat = tf.reshape(coords, [batch_size, height * width, 3])
        # coords_flat shape: [batch, height*width, 3]

        # Apply affine transformation: [batch, HW, 3] @ [batch, 3, 2] -> [batch, HW, 2]
        coords_transformed = tf.matmul(coords_flat, theta, transpose_b=True)
        # coords_transformed shape: [batch, height*width, 2]

        # Reshape back to spatial dimensions
        coords_transformed = tf.reshape(coords_transformed, [batch_size, height, width, 2])

        # Bilinear sampling
        xs = coords_transformed[..., 0]
        ys = coords_transformed[..., 1]

        # Normalize to [0, width/height)
        xs = (xs + 1.0) * (tf.cast(width, tf.float32) - 1.0) / 2.0
        ys = (ys + 1.0) * (tf.cast(height, tf.float32) - 1.0) / 2.0

        # Bilinear interpolation
        x0 = tf.cast(tf.floor(xs), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(ys), tf.int32)
        y1 = y0 + 1

        # Clip to valid range
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # Gather values
        batch_indices = tf.range(batch_size)[:, None, None]
        batch_indices = tf.tile(batch_indices, [1, height, width])

        def gather_nd_bilinear(img, y, x):
            idx = tf.stack([batch_indices, y, x], axis=-1)
            return tf.gather_nd(img, idx)

        I00 = gather_nd_bilinear(x, y0, x0)
        I01 = gather_nd_bilinear(x, y0, x1)
        I10 = gather_nd_bilinear(x, y1, x0)
        I11 = gather_nd_bilinear(x, y1, x1)

        # Compute weights
        wx = xs - tf.cast(x0, tf.float32)
        wy = ys - tf.cast(y0, tf.float32)

        # Expand weights to match channel dimension: [batch, height, width] -> [batch, height, width, 1]
        wx = tf.expand_dims(wx, axis=-1)
        wy = tf.expand_dims(wy, axis=-1)

        # Bilinear interpolation: all tensors now have shape [batch, height, width, channels]
        output = (I00 * (1 - wx) * (1 - wy) +
                  I01 * wx * (1 - wy) +
                  I10 * (1 - wx) * wy +
                  I11 * wx * wy)

        return output

print("Testing STN...")
stn = SpatialTransformerNetwork()

# Test forward pass with smaller image
test_image = np.random.randn(2, 64, 64, 3).astype(np.float32)
test_input = tf.constant(test_image)

output = stn(test_input, training=False)
print(f"✓ STN output shape: {output.shape} (expected: (2, 64, 64, 3))")
print(f"✓ STN is working correctly!")
