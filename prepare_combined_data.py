import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load labels configuration
with open('labels_config.json', 'r') as f:
    labels_config = json.load(f)
print("Labels configuration loaded:", labels_config)

# Load features, labels, sample sizes, and ADSR parameters
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
    print("Features loaded:", features.shape)

with open('features/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    print("Labels loaded:", labels.shape)

with open('features/sample_sizes.pkl', 'rb') as f:
    sample_sizes = pickle.load(f)
    sample_sizes = np.array(sample_sizes)  # Convert to NumPy array
    print("Sample sizes loaded:", sample_sizes.shape)

with open('features/adsr_params.pkl', 'rb') as f:
    adsr_params = pickle.load(f)
    adsr_params = np.array(adsr_params)  # Convert to NumPy array
    print("ADSR parameters loaded:", adsr_params.shape)

# Identify indices of sounds based on labels configuration
label_indices = {label: [i for i, lbl in enumerate(labels) if any(label in str(l).lower() for l in lbl)] for label in labels_config.keys()}
for label, indices in label_indices.items():
    print(f"Found {len(indices)} {label} sounds")

# Extract features, sample sizes, and ADSR parameters based on identified indices and combine them based on specified percentages
combined_features = []
combined_sample_sizes = []
combined_adsr_params = []
for label, percentage in labels_config.items():
    indices = label_indices[label]
    label_features = features[indices]
    label_sample_sizes = sample_sizes[indices]  # Use the precomputed sample sizes
    label_adsr_params = adsr_params[indices]  # Use the precomputed ADSR parameters
    
    combined_features.append(label_features * (percentage / 100.0))
    combined_sample_sizes.extend(label_sample_sizes)  # Collect all sample sizes
    combined_adsr_params.extend(label_adsr_params)  # Collect all ADSR parameters

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

# Create a directory called 'combined_features' to save the .pkl files
output_dir = 'combined_features'
os.makedirs(output_dir, exist_ok=True)

# Save the prepared combined features, scaler, PCA, sample sizes, and ADSR parameters
with open(os.path.join(output_dir, 'combined_features.pkl'), 'wb') as f:
    pickle.dump(combined_features, f)
with open(os.path.join(output_dir, 'combined_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(output_dir, 'combined_pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)
with open(os.path.join(output_dir, 'sample_sizes.pkl'), 'wb') as f:
    pickle.dump(combined_sample_sizes, f)  # Save the combined sample sizes
with open(os.path.join(output_dir, 'adsr_params.pkl'), 'wb') as f:
    pickle.dump(combined_adsr_params, f)  # Save the combined ADSR parameters

print(f"Combined features, scaler, PCA, sample sizes, and ADSR parameters saved to '{output_dir}'")