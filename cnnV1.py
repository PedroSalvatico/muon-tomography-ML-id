import pickle
import os 
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
path_pickle = os.path.join(base_dir, "AI_X_Sample_0.pkl")

# Load the list from the file
with open(path_pickle, 'rb') as f:
    X_sample = pickle.load(f)


# X_sample rotations
# print("X_sample rotations: should display same 3d channel vector")
# print(X_sample[56, 0, 0, 0])  # Original simulation 0, position (0,0,0)
# print(X_sample[57, 0, 1, 0])  # Rotated version of simulation 0, position (1,1,1)
# print(X_sample[58, 1, 1, 0])  # Original simulation 1
# print(X_sample[59, 1, 0, 0])  # Rotated version of simulation 1
print("")
# print(X_sample[4, 0, 0, 0])  # Original simulation 0, position (0,0,0)
# print(X_sample[5, 0, 1, 0])  # Rotated version of simulation 0, position (1,1,1)
# print(X_sample[6, 1, 1, 0])  # Original simulation 1
# print(X_sample[7, 1, 0, 0])  # Rotated version of simulation 1


base_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(base_dir, "log_file_list.txt")

with open(log_file_path, 'r') as file:
    lines = file.read().splitlines()  # Automatically removes '\n'

# For each simulation (2x2x2 grid), create a 2x2x2 label array
# Example materials: 0=air, 1=aluminum, 2=lead, etc.
# 1. Initialize label array correctly (4 rotations per simulation)
num_simulations = len(lines)  # Your original simulation count
y_labels = np.zeros((4 * num_simulations, 2, 2, 2), dtype=np.uint8)

# 2. Define voxel order matching your physical coordinates
voxel_order = [
    (0, 1, 0),  # (-2, 2, 0)
    (1, 1, 0),   # (2, 2, 0)
    (1, 0, 0),   # (2, -2, 0)
    (0, 0, 0),   # (-2, -2, 0)
    (0, 1, 1),   # (-2, 2, 4)
    (1, 1, 1),    # (2, 2, 4)
    (1, 0, 1),    # (2, -2, 4)
    (0, 0, 1)     # (-2, -2, 4)
]

def rotate(config):
    """Generate all rotations of a configuration"""
    A, B, C, D, E, F, G, H = config
    rot1 = [D, A, B, C, H, E, F, G]
    rot2 = [C, D, A, B, G, H, E, F]
    rot3 = [B, C, D, A, F, G, H, E]
    return [rot1, rot2, rot3]

# 3. Corrected material assignment function
def get_simulation_materials(n):
    """Assign materials to all rotations of simulation n"""
    materials = [int(lines[n][31]), int(lines[n][32]), 
                 int(lines[n][33]), int(lines[n][34]),
                 int(lines[n][35]), int(lines[n][36]),
                 int(lines[n][37]), int(lines[n][38])]
    first_rotation, second_rotation, third_rotation = rotate(materials)

    # Base simulation (rotation 0)
    for idx, (i, j, k) in enumerate(voxel_order):
        y_labels[4* n, i, j, k] = materials[idx]
        y_labels[4* n + 1, i, j, k] = first_rotation[idx]
        y_labels[4* n + 2, i, j, k] = second_rotation[idx]
        y_labels[4*n +3, i, j, k] = third_rotation[idx]

    
    # Create rotated versions (rotations 1-3)
    # for rot in range(1, 4):
    #     rotated_materials = np.rot90(
    #         y_labels[4*n].reshape(2,2,2), 
    #         k=rot, 
    #         axes=(0,1)
    #     ).flatten()
        
    #     for idx, (i, j, k) in enumerate(voxel_order):
    #         y_labels[4*n + rot, i, j, k] = rotated_materials[idx]

# 4. Apply to all simulations
for sim in range(num_simulations):
    get_simulation_materials(sim)

# 5. Verification
print("Checking Cube configs: (should give the file id in order) ")
# print(y_labels[56,0,1,0])
# print(y_labels[56,1,1,0])
# print(y_labels[56,1,0,0])
# print(y_labels[56,0,0,0])
# print("")
# print(y_labels[56,0,1,1])
# print(y_labels[56,1,1,1])
# print(y_labels[56,1,0,1])
# print(y_labels[56,0,0,1])

# print("Checking rotations: (should all be equal) ")
# print(y_labels[56,0,1,0])
# print(y_labels[57,1,1,0])
# print(y_labels[58,1,0,0])
# print(y_labels[59,0,0,0])
# print("")


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Your existing arrays:
# X_sample: shape (num_simulations, 2, 2, 2, num_channels)
# y_labels: shape (num_simulations, 2, 2, 2)

# Shift labels from [1,2,3] to [0,1,2]
y_labels = y_labels - 1

# One-hot encode labels voxel-wise
y_onehot = to_categorical(y_labels, num_classes=3)  # Shape: (N, 2, 2, 2, 3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_onehot, test_size=0.2, random_state=42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam

input_shape = (2, 2, 2, X_sample.shape[-1])

inp = Input(shape=input_shape)
x = Conv3D(32, kernel_size=2, activation='relu')(inp)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(8 * 3, activation='softmax')(x)  # 8 voxels, 3 classes each
out = Reshape((2, 2, 2, 3))(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.2%}")

y_pred = model.predict(X_test)                      # shape: (N, 2, 2, 2, 3)
y_pred_labels = np.argmax(y_pred, axis=-1)          # shape: (N, 2, 2, 2)
y_true_labels = np.argmax(y_test, axis=-1)          # same shape

# Optional: flatten everything to compare confusion
from sklearn.metrics import confusion_matrix, classification_report

y_pred_flat = y_pred_labels.flatten()
y_true_flat = y_true_labels.flatten()

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_flat, y_pred_flat))

print("\nClassification Report:")
print(classification_report(y_true_flat, y_pred_flat, digits=3))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the true and predicted labels (already calculated in your code)
# y_pred_labels = np.argmax(y_pred, axis=-1)  # shape: (N, 2, 2, 2)
# y_true_labels = np.argmax(y_test, axis=-1)  # same shape

# Flatten the predictions and true labels for the confusion matrix
y_true_flat = y_true_labels.flatten()
y_pred_flat = y_pred_labels.flatten()

# Create confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)
labels = ['Aluminum', 'Copper', 'Tungsten']  # Or whatever your classes are (0,1,2)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
            xticklabels=labels, yticklabels=labels, 
            annot_kws={'size': 18})

plt.xlabel('Predicted Label', fontsize=17, labelpad=20)
plt.ylabel('True Label', fontsize=17, labelpad=20)
plt.xticks(fontsize=14, fontweight='bold', family='Arial')
plt.yticks(fontsize=14, fontweight='bold', family='Arial')
plt.title('Confusion Matrix (CNN Classifier)', fontsize=20)
plt.tight_layout()
plt.show()