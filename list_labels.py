import numpy as np
import pickle

# Load features and labels
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
    print("Features loaded:", features.shape)

with open('features/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    print("Labels loaded:", labels.shape)

# Get a list of all unique labels
unique_labels = list(set(tuple(label) for label in labels))
print("Unique labels:")
for label in unique_labels:
    print(label)