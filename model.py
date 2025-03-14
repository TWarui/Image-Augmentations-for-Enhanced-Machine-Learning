import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# Step 1: Load the CIFAR-10 Dataset (Manually from local files)
# =============================================================================

def unpickle(file):
    """Loads a CIFAR-10 batch file and returns a dictionary."""
    with open(file, 'rb') as fo:
        dict_obj = pickle.load(fo, encoding='bytes')
    return dict_obj

# Use a raw string for Windows paths so backslashes are not misinterpreted
dataset_path = r"C:\Users\Hp\Documents\SCHOOL\MULTIMEDIA APPLICATIONS\ASSIGNMENT_1\Image-Augmentations-for-Enhanced-Machine-Learning\data\cifar-10-python\cifar-10-batches-py"

# --- Load Training Data ---
X_train = []
y_train = []
for i in range(1, 6):  # data_batch_1 to data_batch_5
    batch = unpickle(os.path.join(dataset_path, f"data_batch_{i}"))
    X_train.append(batch[b'data'])
    y_train.extend(batch[b'labels'])
X_train = np.vstack(X_train)  # Shape becomes (50000, 3072)
y_train = np.array(y_train)     # Shape (50000,)

# Reshape each row into a 32x32 RGB image:
# First reshape to (50000, 3, 32, 32), then transpose to (50000, 32, 32, 3)
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0

# --- Load Test Data ---
test_batch = unpickle(os.path.join(dataset_path, "test_batch"))
X_test = test_batch[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
y_test = np.array(test_batch[b'labels'])

# Convert labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# =============================================================================
# Step 2: Define a Function to Create the CNN Model
# =============================================================================

def create_model():
    """Creates and compiles a simple CNN model."""
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# =============================================================================
# Step 3: Train Baseline Model (Without Augmentation)
# =============================================================================

print("Training baseline model (no augmentation)...")
model_baseline = create_model()
history_baseline = model_baseline.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

test_loss_baseline, test_acc_baseline = model_baseline.evaluate(X_test, y_test, verbose=2)
print(f"Baseline Model Test Accuracy: {test_acc_baseline * 100:.2f}%")

# =============================================================================
# Step 4: Train Augmented Model (With Data Augmentation)
# =============================================================================

# Define the ImageDataGenerator for augmentation with several techniques:
datagen = ImageDataGenerator(
    rotation_range=20,        # Randomly rotate images by 20 degrees
    width_shift_range=0.2,    # Randomly shift images horizontally (20% of width)
    height_shift_range=0.2,   # Randomly shift images vertically (20% of height)
    horizontal_flip=True,     # Randomly flip images horizontally
    zoom_range=0.2            # Randomly zoom in on images
)
# Compute any statistics required by the generator (not strictly necessary here)
datagen.fit(X_train)

print("Training augmented model...")
model_augmented = create_model()
history_augmented = model_augmented.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=len(X_train) // 64,
    epochs=10,
    validation_data=(X_test, y_test)
)

test_loss_aug, test_acc_aug = model_augmented.evaluate(X_test, y_test, verbose=2)
print(f"Augmented Model Test Accuracy: {test_acc_aug * 100:.2f}%")

# =============================================================================
# Step 5: Compare the Performance of Both Models
# =============================================================================

plt.figure(figsize=(14, 6))

# ----- Accuracy Comparison -----
plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['accuracy'], label='Baseline Train Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Baseline Validation Accuracy')
plt.plot(history_augmented.history['accuracy'], label='Augmented Train Accuracy')
plt.plot(history_augmented.history['val_accuracy'], label='Augmented Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy Comparison")

# ----- Loss Comparison -----
plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['loss'], label='Baseline Train Loss')
plt.plot(history_baseline.history['val_loss'], label='Baseline Validation Loss')
plt.plot(history_augmented.history['loss'], label='Augmented Train Loss')
plt.plot(history_augmented.history['val_loss'], label='Augmented Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Comparison")

plt.tight_layout()
plt.show()

# =============================================================================
# Summary:
# - The baseline model was trained on the original dataset.
# - The augmented model was trained using real-time data augmentation.
# - By comparing the training and validation accuracy/loss, you can observe differences
#   in performance. For example, data augmentation might reduce overfitting by improving
#   validation performance even if training accuracy is slightly lower.
# =============================================================================
