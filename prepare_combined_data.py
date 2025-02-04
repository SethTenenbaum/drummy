import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load labels configuration
with open('labels_config.json', 'r') as f:
    labels_config = json.load(f)
print("Labels configuration loaded:", labels_config)

# Load features and labels
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
    print("Features loaded:", features.shape)

with open('features/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    print("Labels loaded:", labels.shape)

# Identify indices of sounds based on labels configuration
label_indices = {label: [i for i, lbl in enumerate(labels) if label in lbl.lower()] for label in labels_config.keys()}
for label, indices in label_indices.items():
    print(f"Found {len(indices)} {label} sounds")

# Extract features based on identified indices and combine them based on specified percentages
combined_features = []
for label, percentage in labels_config.items():
    indices = label_indices[label]
    label_features = features[indices]
    combined_features.append(label_features * (percentage / 100.0))

# Combine all features into a single array
combined_features = np.vstack(combined_features)
print("Combined features shape:", combined_features.shape)

# Normalize features
scaler = StandardScaler()
combined_features = scaler.fit_transform(combined_features)

# Apply PCA to reduce the number of features
pca = PCA(n_components=20)  # Adjust the number of components as needed
combined_features = pca.fit_transform(combined_features)
print("Combined features after PCA:", combined_features.shape)

# Save the prepared combined features, scaler, and PCA
with open('combined_features.pkl', 'wb') as f:
    pickle.dump(combined_features, f)
with open('combined_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('combined_pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
print("Combined features, scaler, and PCA saved")