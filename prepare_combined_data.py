import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load features and labels
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
    print("Features loaded:", features.shape)

with open('features/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    print("Labels loaded:", labels.shape)

# Identify indices of snare and cymbal sounds
snare_indices = [i for i, label in enumerate(labels) if 'snare' in label.lower()]
cymbal_indices = [i for i, label in enumerate(labels) if 'cymbal' in label.lower()]
print(f"Found {len(snare_indices)} snare sounds")
print(f"Found {len(cymbal_indices)} cymbal sounds")

# Extract snare and cymbal features
snare_features = features[snare_indices]
cymbal_features = features[cymbal_indices]

# Combine snare and cymbal features based on specified percentages
snare_percentage = 0.8
cymbal_percentage = 0.2
combined_features = np.vstack([
    snare_features * snare_percentage,
    cymbal_features * cymbal_percentage
])
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