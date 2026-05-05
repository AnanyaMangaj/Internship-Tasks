import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Layer, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
import tensorflow as tf

print("="*60)
print("VARIATIONAL AUTOENCODER (VAE)")
print("="*60)

# Load CIFAR10 data
print("\n[1/5] Loading CIFAR10 dataset...")
(X_train, _), (X_test, _) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(f"✓ Loaded {X_train.shape[0]} training images, {X_test.shape[0]} test images")

# Hyperparameters
latent_dim = 32
input_shape = (32, 32, 3)

# Sampling layer
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build Encoder
print("\n[2/5] Building Encoder...")
encoder_inputs = Input(shape=input_shape)
x = Flatten()(encoder_inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
print("✓ Encoder built")

# Build Decoder
print("\n[3/5] Building Decoder...")
latent_inputs = Input(shape=(latent_dim,))
x = Dense(128, activation='relu')(latent_inputs)
x = Dense(256, activation='relu')(x)
x = Dense(32*32*3, activation='sigmoid')(x)
decoder_outputs = Reshape(input_shape)(x)

decoder = Model(latent_inputs, decoder_outputs, name='decoder')
print("✓ Decoder built")

# Build VAE
print("\n[4/5] Building VAE...")

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def compute_losses(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - reconstruction), axis=[1, 2, 3])
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
        )
        return reconstruction, reconstruction_loss, kl_loss

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            reconstruction, reconstruction_loss, kl_loss = self.compute_losses(x)
            loss = reconstruction_loss + kl_loss
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        _, reconstruction_loss, kl_loss = self.compute_losses(x)
        loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder, name='vae')
vae.compile(optimizer='adam')
print("✓ VAE compiled")

# Train
print("\n[5/5] Training VAE (5 epochs)...\n")
history = vae.fit(X_train, epochs=5, batch_size=128, validation_data=X_test, verbose=1)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Reconstruct images
print("\nReconstructing test images...")
test_images = X_test[:9]
reconstructed = vae.predict(test_images, verbose=0)

# Display original vs reconstructed
fig, axes = plt.subplots(3, 6, figsize=(15, 7))
fig.suptitle('VAE: Original vs Reconstructed Images', fontsize=14, fontweight='bold')

for i in range(9):
    # Original
    axes[i//3, i%3*2].imshow(test_images[i])
    axes[i//3, i%3*2].set_title(f'Original {i+1}')
    axes[i//3, i%3*2].axis('off')
    
    # Reconstructed
    axes[i//3, i%3*2+1].imshow(reconstructed[i])
    axes[i//3, i%3*2+1].set_title(f'Reconstructed {i+1}')
    axes[i//3, i%3*2+1].axis('off')

plt.tight_layout()
plt.savefig('vae_reconstruction.png', dpi=100, bbox_inches='tight')
print("✓ Reconstruction comparison saved as 'vae_reconstruction.png'")
plt.show()

# Generate new images from random latent vectors
print("\nGenerating new images from random latent codes...")
random_latent_vectors = np.random.normal(size=(9, latent_dim))
generated_images = decoder.predict(random_latent_vectors, verbose=0)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
fig.suptitle('Generated Images from Random Latent Vectors', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i])
    ax.set_title(f'Generated {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('vae_generated.png', dpi=100, bbox_inches='tight')
print("✓ Generated images saved as 'vae_generated.png'")
plt.show()

# Print summary
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Encoder Parameters: {encoder.count_params():,}")
print(f"Decoder Parameters: {decoder.count_params():,}")
print(f"VAE Parameters: {vae.count_params():,}")
print(f"Latent Dimension: {latent_dim}")
print(f"Input Shape: {input_shape}")
print("="*60)