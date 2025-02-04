import librosa
import numpy as np
import os
import pickle
import warnings

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=44100)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Initialize feature list and feature names
    feature_list = []
    feature_names = ["MFCCs", "Chroma features", "Spectral Contrast", "Zero-Crossing Rate", "Spectral Centroid", 
                     "Spectral Bandwidth", "Spectral Roll-off", "RMS", "Spectral Flatness", "Spectral Flux"]
    feature_lengths = {"MFCCs": 13, "Chroma features": 12, "Spectral Contrast": 7, "Zero-Crossing Rate": 1, 
                       "Spectral Centroid": 1, "Spectral Bandwidth": 1, "Spectral Roll-off": 1, "RMS": 1, 
                       "Spectral Flatness": 1, "Spectral Flux": 1}
    
    # Function to extract a feature and handle warnings
    def safe_extract(feature_name, feature_func, *args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                return feature_func(*args, **kwargs)
        except Warning as w:
            print(f"Skipping {feature_name} for {file_path} due to warning: {w}")
            return None
        except Exception as e:
            print(f"Error extracting {feature_name} for {file_path}: {e}")
            return None
    
    # Extract features
    mfccs = safe_extract("MFCCs", librosa.feature.mfcc, y=y, sr=sr, n_mfcc=13, n_fft=1024)
    chroma = safe_extract("Chroma features", librosa.feature.chroma_stft, y=y, sr=sr, n_fft=1024)
    spectral_contrast = safe_extract("Spectral Contrast", librosa.feature.spectral_contrast, y=y, sr=sr, n_fft=1024)
    zcr = safe_extract("Zero-Crossing Rate", librosa.feature.zero_crossing_rate, y)
    spectral_centroid = safe_extract("Spectral Centroid", librosa.feature.spectral_centroid, y=y, sr=sr, n_fft=1024)
    spectral_bandwidth = safe_extract("Spectral Bandwidth", librosa.feature.spectral_bandwidth, y=y, sr=sr, n_fft=1024)
    spectral_rolloff = safe_extract("Spectral Roll-off", librosa.feature.spectral_rolloff, y=y, sr=sr, n_fft=1024)
    rms = safe_extract("RMS", librosa.feature.rms, y=y)
    spectral_flatness = safe_extract("Spectral Flatness", librosa.feature.spectral_flatness, y=y)
    spectral_flux = safe_extract("Spectral Flux", lambda y: np.diff(np.abs(librosa.stft(y, n_fft=1024)), axis=1).mean(axis=1), y)
    
    # Combine all features into a single feature vector
    features = []
    for feature, name in zip([mfccs, chroma, spectral_contrast, zcr, spectral_centroid, spectral_bandwidth, 
                              spectral_rolloff, rms, spectral_flatness, spectral_flux], feature_names):
        if feature is not None:
            features.append(np.mean(feature.T, axis=0))
        else:
            print(f"Filling missing {name} with zeros for {file_path}")
            features.append(np.zeros(feature_lengths[name]))  # Use predefined lengths for each feature
    
    if features:
        features = np.hstack(features)
    else:
        features = None
    
    return features

sample_dir = 'samples/'
features = []
labels = []
skipped_files = []

# Debug print to check if the directory is being read correctly
print(f"Reading files from directory: {sample_dir}")

# Get the total number of .wav files
total_files = sum(len([f for f in files if f.endswith('.wav')]) for _, _, files in os.walk(sample_dir))
processed_files = 0

# Recursively traverse the directory structure
for root, dirs, files in os.walk(sample_dir):
    for file_name in files:
        if file_name.endswith('.wav'):
            file_path = os.path.join(root, file_name)
            print(f"Processing file: {file_path}")
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                
                # Use filename and folder structure as part of the label
                relative_path = os.path.relpath(file_path, sample_dir)
                folder_labels = os.path.dirname(relative_path).split(os.sep)
                file_label = os.path.splitext(file_name)[0]
                combined_label = folder_labels + [file_label]
                labels.append(combined_label)  # Use folder structure and filename as labels
            else:
                skipped_files.append(file_path)
            processed_files += 1
            percentage_completed = (processed_files / total_files) * 100
            print(f"Processed {processed_files}/{total_files} files ({percentage_completed:.2f}% completed)")

# Debug print to check the lengths of all feature vectors
feature_lengths_set = set(len(f) for f in features)
print(f"Feature vector lengths: {feature_lengths_set}")

# Print skipped files
print(f"Skipped files: {len(skipped_files)}")
for skipped_file in skipped_files:
    print(f"Skipped file: {skipped_file}")

# Pad labels to ensure they all have the same length
max_label_length = max(len(label) for label in labels)
padded_labels = [label + [''] * (max_label_length - len(label)) for label in labels]

# Convert features and labels to NumPy arrays
try:
    features = np.array(features)
    labels = np.array(padded_labels)
except ValueError as e:
    print(f"Error converting features to NumPy array: {e}")
    features = None
    labels = None

if features is not None and labels is not None:
    print("Extracted features shape:", features.shape)
    print("Extracted labels shape:", labels.shape)

    # Create a folder called 'features' to save the .pkl files
    output_dir = 'features'
    os.makedirs(output_dir, exist_ok=True)

    # Save features and labels to files
    with open(os.path.join(output_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)

    with open(os.path.join(output_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print(f"Features and labels saved to {output_dir}")
else:
    print("Failed to convert features and labels to NumPy arrays.")