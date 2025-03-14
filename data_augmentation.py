import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf


#Fuction that loads CIFAR-10 batch files and returns a dictionary
def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Loading a single batch
#define dataset path
dataset_path = r"C:\Users\Hp\Documents\SCHOOL\MULTIMEDIA APPLICATIONS\ASSIGNMENT_1\Image-Augmentations-for-Enhanced-Machine-Learning\data\cifar-10-python\cifar-10-batches-py"

#load one batch
batch_1 = unpickle(os.path.join(dataset_path, "data_batch_1"))
#extract data and labels
data = batch_1[b'data']
labels = batch_1[b'labels']
print("Data shape:", data.shape)  # Should print (10000, 3072)
print("Labels shape:", len(labels))  # Should print 10000

#reshape data into IMage FOrmat
#reshape to (10000, 32, 32, 3) for images
data = data.reshape(10000, 3, 32, 32 ).transpose(0,2,3,1) #convert to (10000, 32, 3,3)

#Load all training batches
X_train = []
y_train = []

# Loop through data_batch_1 to data_batch_5
for i in range(1,6): 
   batch = unpickle(os.path.join(dataset_path, f"data_batch_{i}"))
   X_train.append(batch[b'data'])
   y_train.extend(batch[b'labels'])  # Append labels

# Convert to numpy arrays
X_train = np.vstack(X_train)  # Shape (50000, 3072)
y_train = np.array(y_train)   # Shape (50000,)

# Reshape X_train to (50000, 32, 32, 3)
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)

#load test data 
test_batch = unpickle(os.path.join(dataset_path, "test_batch"))
X_test = test_batch[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
y_test = np.array(test_batch[b'labels'])

# Load class names
meta = unpickle(os.path.join(dataset_path, "batches.meta"))
label_names = [name.decode('utf-8') for name in meta[b'label_names']]

print(label_names)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#show 5 random images
# fig, axes = plt.subplots(1,5, figsize=(10,5))

# for i, ax in enumerate(axes):
#     idx = np.random.randint(0, X_train.shape[0]) #pick a random image
#     ax.imshow(X_train[idx])
#     ax.set_title(label_names[y_train[idx]])
#     ax.axis('off')
# plt.show

#select 5 random images
random_indices = random.sample(range(len(X_train)), 5)
selected_images = X_train[random_indices]
selected_labels = [label_names[y_train[i]] for i in random_indices]

# Image augmentation techniques using tensorflow
augmentations = {
    "Random Flip": tf.keras.layers.RandomFlip("horizontal"),
    "Random Rotation": tf.keras.layers.RandomRotation(0.2),
    "Random Zoom": tf.keras.layers.RandomZoom(0.2),
    "Random Contrast": tf.keras.layers.RandomContrast(0.2),
    "Random Translation": tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
}

#apply augmentation and visualize results 
fig, axes = plt.subplots(5,2, figsize=(10, 12))

for i, (aug_name, aug_layer) in enumerate(augmentations.items()):
    original = selected_images[i] / 255.0  # Normalize image for display
    augmented = aug_layer(tf.expand_dims(original, axis=0))  # Apply augmentation
    
    # Plot original image
    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f"Original: {selected_labels[i]}")
    axes[i, 0].axis("off")
    
    # Plot augmented image with augmentation type in the title
    axes[i, 1].imshow(augmented[0].numpy())
    axes[i, 1].set_title(f"Augmented ({aug_name})")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()