#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from PIL import Image, ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

TRAIN_PATH = './data/train'
HEIGHT, WIDTH = 400, 400
SECRET_SIZE = 256
BATCH_SIZE = 4
NUM_STEPS = 280000
LEARNING_RATE = 0.0001

print("Loading training images...", flush=True)
img_files = (glob.glob(os.path.join(TRAIN_PATH, '*.jpg')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.png')) +
             glob.glob(os.path.join(TRAIN_PATH, '*.webp')))
if not img_files:
    raise FileNotFoundError(f"No images in {TRAIN_PATH}")
print(f"Found {len(img_files)} images", flush=True)

def get_img_batch(batch_size=BATCH_SIZE):
    batch_cover = []
    batch_secret = []

    for _ in range(batch_size):
        try:
            img_path = np.random.choice(img_files)
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, (HEIGHT, WIDTH))
            img = np.array(img, dtype=np.float32) / 255.0
        except:
            img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        batch_cover.append(img)

        secret = np.random.binomial(1, 0.5, SECRET_SIZE).astype(np.float32)
        batch_secret.append(secret)

    return np.array(batch_cover), np.array(batch_secret)

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

class SpatialTransformerNetwork(keras.Model):
    """Spatial Transformer Network for geometric robustness"""
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

class StegaStampDecoder(keras.Model):
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


print("Building models...", flush=True)
encoder = StegaStampEncoder()
decoder = StegaStampDecoder()
critic = StegaStampCritic()

enc_opt = keras.optimizers.Adam(LEARNING_RATE)
dec_opt = keras.optimizers.Adam(LEARNING_RATE)
crit_opt = keras.optimizers.Adam(LEARNING_RATE * 0.5)  # Slightly lower LR for critic stability

# Load pre-trained VGG16 for LPIPS
# Lazy-load VGG16 only when LPIPS is needed (Phase 2+)
lpips_model = None

def lpips_loss(image1, image2):
    """Compute LPIPS perceptual loss using VGG16 features"""
    global lpips_model

    # Lazy load VGG16 on first use
    if lpips_model is None:
        print("Loading VGG16 for perceptual loss (first use in Phase 2+)...", flush=True)
        vgg16_full = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        vgg16_full.trainable = False

        layer_names = ['block1_pool', 'block2_pool', 'block3_pool']
        lpips_model = keras.Model(
            inputs=vgg16_full.input,
            outputs=[vgg16_full.get_layer(name).output for name in layer_names]
        )
        print("VGG16 loaded successfully", flush=True)

    # Normalize images to [0, 1] for VGG16
    img1_norm = (image1 + 0.5)  # Convert from [-0.5, 0.5] to [0, 1]
    img2_norm = (image2 + 0.5)

    # Ensure 3D spatial dimensions
    if len(img1_norm.shape) == 3:
        img1_norm = tf.expand_dims(img1_norm, 0)
    if len(img2_norm.shape) == 3:
        img2_norm = tf.expand_dims(img2_norm, 0)

    # Resize to VGG16 input size (224x224) if needed
    if tf.shape(img1_norm)[1] != 224:
        img1_norm = tf.image.resize(img1_norm, (224, 224))
        img2_norm = tf.image.resize(img2_norm, (224, 224))

    # Get feature maps
    features1 = lpips_model(img1_norm * 255.0, training=False)  # VGG expects [0, 255]
    features2 = lpips_model(img2_norm * 255.0, training=False)

    # Compute L2 distance for each layer with weights
    losses = []
    weights = [0.1, 0.1, 1.0]
    for feat1, feat2, w in zip(features1, features2, weights):
        loss = tf.reduce_mean(tf.square(feat1 - feat2)) * w
        losses.append(loss)

    return tf.reduce_mean(losses)

def train_step(images, secrets, secret_scale, l2_scale, critic_scale, lpips_scale):
    with tf.GradientTape(persistent=True) as tape:
        residual = encoder([secrets, images], training=True)
        encoded = images + residual

        decoded = decoder(encoded, training=True)

        # Compute losses
        secret_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=secrets, logits=decoded))
        l2_loss = tf.reduce_mean(tf.square(residual))

        # LPIPS loss
        lpips_loss_val = lpips_loss(images, encoded) if lpips_scale > 0 else tf.constant(0.0)

        # Critic loss
        critic_fake = critic(encoded, training=True)
        critic_real = critic(images, training=True)
        critic_loss = -tf.reduce_mean(tf.math.log(critic_real + 1e-8) + tf.math.log(1 - critic_fake + 1e-8))

        # Total loss for encoder/decoder (critic wants them adversarial)
        enc_dec_loss = (secret_scale * secret_loss +
                        l2_scale * l2_loss +
                        lpips_scale * lpips_loss_val +
                        critic_scale * (-tf.reduce_mean(tf.math.log(critic_fake + 1e-8))))  # Fooling critic

    # Update encoder and decoder
    enc_grads = tape.gradient(enc_dec_loss, encoder.trainable_variables)
    dec_grads = tape.gradient(enc_dec_loss, decoder.trainable_variables)

    enc_opt.apply_gradients(zip(enc_grads, encoder.trainable_variables))
    dec_opt.apply_gradients(zip(dec_grads, decoder.trainable_variables))

    # Update critic
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    crit_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))

    return secret_loss, l2_loss, lpips_loss_val, critic_loss, residual

def get_loss_scales(step, total_steps):
    """
    Phase 1 (0-10%): Secret-only learning, no critic/LPIPS
    Phase 2 (10%-40%): Introduce critic and LPIPS gradually
    Phase 3 (40%-100%): Full multi-loss training
    """
    secret_scale = tf.constant(1.5, dtype=tf.float32)
    phase1_end = int(total_steps * 0.1)
    phase2_end = int(total_steps * 0.4)

    # L2 loss schedule
    if step < phase1_end:
        progress = step / phase1_end
        l2_scale = tf.constant(float(progress * 0.5), dtype=tf.float32)
    elif step < phase2_end:
        progress = (step - phase1_end) / (phase2_end - phase1_end)
        l2_scale = tf.constant(float(0.5 + progress * 1.5), dtype=tf.float32)
    else:
        l2_scale = tf.constant(2.0, dtype=tf.float32)

    # Critic loss schedule: 0 in phase 1, ramp 0->0.1 in phase 2, full in phase 3
    if step < phase1_end:
        critic_scale = tf.constant(0.0, dtype=tf.float32)
    elif step < phase2_end:
        progress = (step - phase1_end) / (phase2_end - phase1_end)
        critic_scale = tf.constant(float(progress * 0.1), dtype=tf.float32)
    else:
        critic_scale = tf.constant(0.1, dtype=tf.float32)

    # LPIPS loss schedule: 0 in phase 1, ramp 0->0.5 in phase 2, full in phase 3
    if step < phase1_end:
        lpips_scale = tf.constant(0.0, dtype=tf.float32)
    elif step < phase2_end:
        progress = (step - phase1_end) / (phase2_end - phase1_end)
        lpips_scale = tf.constant(float(progress * 0.5), dtype=tf.float32)
    else:
        lpips_scale = tf.constant(0.5, dtype=tf.float32)

    return secret_scale, l2_scale, critic_scale, lpips_scale

print(f"Training 256-bit secrets for {NUM_STEPS} steps...", flush=True)
phase1_end = int(NUM_STEPS * 0.1)
phase2_end = int(NUM_STEPS * 0.4)
print(f"Phase 1 (secret-ramp): steps 0-{phase1_end} | Phase 2 (ramp): {phase1_end+1}-{phase2_end} | Phase 3 (full): {phase2_end+1}-{NUM_STEPS}", flush=True)

for step in range(NUM_STEPS):
    images, secrets = get_img_batch(BATCH_SIZE)
    images = tf.constant(images)
    secrets = tf.constant(secrets)

    secret_scale, l2_scale, critic_scale, lpips_scale = get_loss_scales(step, NUM_STEPS)
    sec_loss, l2_loss, lpips_loss_val, crit_loss, residual = train_step(images, secrets, secret_scale, l2_scale, critic_scale, lpips_scale)

    if (step + 1) % 50 == 0:
        res_mean = tf.reduce_mean(tf.abs(residual)).numpy()
        res_max = tf.reduce_max(tf.abs(residual)).numpy()
        if step < phase1_end:
            phase = "P1"
        elif step < phase2_end:
            phase = "P2"
        else:
            phase = "P3"
        total_loss = float(secret_scale * sec_loss + l2_scale * l2_loss + lpips_scale * lpips_loss_val + critic_scale * crit_loss)
        print(f"[{phase}] Step {step+1:6d}/{NUM_STEPS} | Secret:{float(sec_loss):.4f} L2:{float(l2_loss):.4f} LPIPS:{float(lpips_loss_val):.4f} Critic:{float(crit_loss):.4f} | Residual:{res_mean:.6f} | Scales: L2={float(l2_scale):.3f} Critic={float(critic_scale):.3f} LPIPS={float(lpips_scale):.3f}", flush=True)

    if step < 5:
        res_mean = tf.reduce_mean(tf.abs(residual)).numpy()
        print(f"[INIT Step {step}] Residual:{res_mean:.8f} L2:{float(l2_loss):.8f} Secret:{float(sec_loss):.4f}", flush=True)

    if (step + 1) % 10000 == 0:
        encoder.save(f'encoder_256bit_step_{step+1}.keras')
        decoder.save(f'decoder_256bit_step_{step+1}.keras')
        critic.save(f'critic_256bit_step_{step+1}.keras')
        checkpoint_mb = os.path.getsize(f'encoder_256bit_step_{step+1}.keras') / (1024*1024)
        print(f"✓ CHECKPOINT saved: step {step+1} ({checkpoint_mb:.0f}MB) | Secret:{float(sec_loss):.4f} L2:{float(l2_loss):.4f} Critic:{float(crit_loss):.4f}", flush=True)

encoder.save('encoder_256bit_final.keras')
decoder.save('decoder_256bit_final.keras')
critic.save('critic_256bit_final.keras')
print("✓ Training complete! All models saved.", flush=True)
