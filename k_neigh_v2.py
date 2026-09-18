import pickle
import os 
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
path_pickle = os.path.join(base_dir, "AI_X_Sample_0.pkl")

# Load the list from the file
with open(path_pickle, 'rb') as f:
    X_sample = pickle.load(f)




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

    
   
# 4. Apply to all simulations
for sim in range(num_simulations):
    get_simulation_materials(sim)

#######################################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Flatten data
X = X_sample.reshape(-1, 3)     # or 2, or whatever number of channels you want
y = y_labels.reshape(-1) - 1    # Adjust labels if needed

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)  # You can try other values like 3, 7, etc.
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))


# VISUALIZATION


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Replace these with your actual y_test and y_pred
# y_test = ...
# y_pred = ...

cm = confusion_matrix(y_test, y_pred)
labels = ['Aluminum', 'Copper', 'Tungsten']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=labels, yticklabels=labels, annot_kws={'size': 18})


plt.xlabel('Predicted Label', fontsize=17, labelpad=20)
plt.ylabel('True Label', fontsize=17, labelpad=20)
plt.xticks(fontsize=14, fontweight='bold', family='Arial')
plt.yticks(fontsize=14, fontweight='bold', family='Arial')
plt.title('Confusion Matrix (KNN Classifier)', fontsize=20)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Assume X and y already exist and y has 3 classes: 0, 1, 2
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Get predicted probabilities
y_score = knn.predict_proba(X_test)  # This returns a list of arrays

# Flatten the probabilities into shape (n_samples, n_classes)
y_score = np.stack([y_score[i][:, 1] for i in range(n_classes)], axis=1)

# Plot ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure(figsize=(6, 5))
colors = ['blue', 'green', 'red']
mats = ['Aluminum', 'Copper', 'Tungsten']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{mats[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.legend(loc="lower right",  fontsize=22)
plt.grid(True)
plt.tight_layout()


plt.xlabel('False Positive Rate', fontsize=17)
plt.ylabel('True Positive Rate', fontsize=17)
plt.xticks(fontsize=14, fontweight='bold', family='Arial')
plt.yticks(fontsize=14, fontweight='bold', family='Arial')
# plt.title('ROC Curve - KNN (One-vs-Rest)', fontsize=17)


plt.show()