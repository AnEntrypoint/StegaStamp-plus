import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 256

class StegaStampEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.secret_dense = keras.layers.Dense(7500, activation='relu', kernel_initializer='he_normal')
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv4 = keras.layers.Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv5 = keras.layers.Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.up6 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
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
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(keras.layers.UpSampling2D(size=(2, 2))(conv5))
        merge6 = keras.layers.Concatenate()([conv4, up6])
        conv6 = self.conv6(merge6)
        up7 = self.up7(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.Concatenate()([conv3, up7])
        conv7 = self.conv7(merge7)
        up8 = self.up8(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.Concatenate()([conv2, up8])
        conv8 = self.conv8(merge8)
        up9 = self.up9(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.Concatenate()([conv1, up9, x])
        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        residual = self.residual(conv10)
        return residual

class StrongerDecoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.decoder = keras.Sequential([
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(SECRET_SIZE)
        ])

    def call(self, image):
        image = image - 0.5
        return self.decoder(image)

print("Testing STRONGER decoder with improved architecture...")
encoder = StegaStampEncoder()
decoder = StrongerDecoder()

# Create test data
images = np.random.uniform(-0.5, 0.5, (2, HEIGHT, WIDTH, 3)).astype(np.float32)
images_tf = tf.constant(images)

# Test with different secrets
secrets_zeros = np.zeros((2, SECRET_SIZE), dtype=np.float32)
secrets_ones = np.ones((2, SECRET_SIZE), dtype=np.float32)

residual_zeros = encoder([tf.constant(secrets_zeros), images_tf], training=True)
residual_ones = encoder([tf.constant(secrets_ones), images_tf], training=True)

# Test decoder
encoded_zeros = images + residual_zeros.numpy()
encoded_ones = images + residual_ones.numpy()

decoded_zeros = decoder(tf.constant(encoded_zeros), training=True).numpy()
decoded_ones = decoder(tf.constant(encoded_ones), training=True).numpy()

print(f"✓ Decoder output for zeros secret: mean={decoded_zeros.mean():.4f}, std={decoded_zeros.std():.4f}")
print(f"✓ Decoder output for ones secret:  mean={decoded_ones.mean():.4f}, std={decoded_ones.std():.4f}")
print(f"✓ Decoded output difference: {np.mean(np.abs(decoded_zeros - decoded_ones)):.6f}")

# Test loss
secret_loss_zeros = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets_zeros, logits=decoded_zeros))
secret_loss_ones = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets_ones, logits=decoded_ones))

print(f"\n✓ Secret loss for zeros: {secret_loss_zeros.numpy():.6f} (target: 0.0)")
print(f"✓ Secret loss for ones:  {secret_loss_ones.numpy():.6f} (target: 0.0)")
print(f"✓ Random baseline: 0.693147")

if secret_loss_zeros.numpy() < 0.69 or secret_loss_ones.numpy() < 0.69:
    print("\n✓✓✓ PROMISING! Loss below random baseline!")
else:
    print("\n⚠️  Still at random baseline (but decoder has more capacity now)")
