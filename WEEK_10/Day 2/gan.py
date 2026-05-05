import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.layers import Reshape, LeakyReLU, Dropout

print("="*60)
print("GAN - Generative Adversarial Network")
print("="*60)

# Load CIFAR10 dataset
print("\n[1/5] Loading CIFAR10 dataset...")
(x_train, _), (_, _) = cifar10.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]
print(f"✓ Loaded {x_train.shape[0]} training images")

# Define Generator
print("\n[2/5] Building Generator...")
generator = Sequential([
    Dense(256, input_dim=100, activation='relu'),
    Dense(512, activation='relu'),
    Dense(32*32*3, activation='tanh'),
    Reshape((32,32,3))
])
print("✓ Generator built")

# Define Discriminator with Conv2D layers
print("\n[3/5] Building Discriminator...")
discriminator = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
    Dropout(0.3),
    Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same'),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("✓ Discriminator built")

# Create GAN
print("\n[4/5] Creating GAN model...")
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')
print("✓ GAN created")

# Train the GAN
print("\n[5/5] Training GAN (10 epochs)...\n")
batch_size = 64
epochs = 10

for epoch in range(epochs):
    # Train Discriminator on real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    
    # Train Discriminator on fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise, verbose=0)
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    
    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    print(f"Epoch {epoch+1}/{epochs} | D_Loss_Real: {d_loss_real[0]:.4f} | D_Loss_Fake: {d_loss_fake[0]:.4f} | G_Loss: {g_loss:.4f}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Generate and visualize final images
print("\nGenerating final fake images...")
noise = np.random.normal(0, 1, (9, 100))
generated_images = generator.predict(noise, verbose=0)

# Denormalize images
generated_images = (generated_images * 127.5 + 127.5).astype('uint8')

# Display generated images
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
fig.suptitle('Generated Images by GAN', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i])
    ax.set_title(f'Image {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('generated_images.png', dpi=100, bbox_inches='tight')
print("✓ Generated images saved as 'generated_images.png'")
plt.show()

# Print summary
print("\nGenerated Image Shape:", generated_images.shape)
print("Generator Parameters:", generator.count_params())
print("Discriminator Parameters:", discriminator.count_params())