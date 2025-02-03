import librosa
import numpy as np
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

sample_dir = 'samples/'
features = []
labels = []

for file_name in os.listdir(sample_dir):
    if file_name.endswith('.wav'):
        file_path = os.path.join(sample_dir, file_name)
        feature = extract_features(file_path)
        features.append(feature)
        labels.append(file_name.split('_')[0])  # Assuming file names are like 'bass_01.wav'

features = np.array(features)
labels = np.array(labels)

# Save features and labels to files
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)