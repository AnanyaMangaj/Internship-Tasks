import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load CIFAR10 dataset
print("Loading CIFAR10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
print(f"✓ Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples\n")

model = Sequential([
    Flatten(input_shape=(32,32,3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

print("Compiling model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nTraining model...\n")
model.fit(X_train/255, y_train, epochs=5, batch_size=32, validation_split=0.2)

print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(X_test/255, y_test)
print(f"\nTest Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")