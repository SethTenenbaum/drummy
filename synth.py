import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf

# Load the trained model
model = load_model('saved_model/drum_model.keras')

# Load the scaler and PCA
with open('features/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Function to extract features from a new drum sound
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024)
    rms = librosa.feature.rms(y=y)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flux = np.diff(np.abs(librosa.stft(y, n_fft=1024)), axis=1).mean(axis=1)
    
    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0),
        np.mean(zcr.T, axis=0),
        np.mean(spectral_centroid.T, axis=0),
        np.mean(spectral_bandwidth.T, axis=0),
        np.mean(spectral_rolloff.T, axis=0),
        np.mean(rms.T, axis=0),
        np.mean(spectral_flatness.T, axis=0),
        np.mean(spectral_flux.T, axis=0)
    ])
    
    return features

# Function to load a sound file
def load_sound(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    return y, sr

# Function to combine sounds based on specified weights
def combine_sounds(sounds, weights):
    combined_sound = np.zeros_like(sounds[0])
    for sound, weight in zip(sounds, weights):
        combined_sound += weight * sound
    combined_sound = combined_sound / np.max(np.abs(combined_sound))  # Normalize
    return combined_sound

# Load a new drum sound and extract features
file_path = 'path/to/new_drum_sound.wav'
new_features = extract_features(file_path)

# Normalize and apply PCA to the new features
new_features = scaler.transform([new_features])
new_features = pca.transform(new_features)

# Use the model to predict the class of the new drum sound
prediction = model.predict(new_features)
predicted_class = np.argmax(prediction, axis=1)[0]
print(f"Predicted class: {predicted_class}")

# Define the file paths for the drum sounds corresponding to each class
drum_sounds = {
    0: 'path/to/snare.wav',
    1: 'path/to/bass_drum.wav',
    2: 'path/to/highhat.wav',
    3: 'path/to/cymbal.wav'
    # Add more classes and corresponding file paths as needed
}

# Define the percentages for each drum sound (example percentages)
percentages = {
    0: 0.1,  # 10% snare
    1: 0.2,  # 20% bass drum
    2: 0.5,  # 50% highhat
    3: 0.3   # 30% cymbal
}

# Load the drum sounds based on the predicted class and percentages
sounds = []
weights = []
for class_id, percentage in percentages.items():
    if class_id in drum_sounds:
        sound, sr = load_sound(drum_sounds[class_id])
        sounds.append(sound)
        weights.append(percentage)

# Combine the sounds based on the specified weights
combined_sound = combine_sounds(sounds, weights)

# Save the combined sound to a new file
output_file_path = 'combined_drum_sound.wav'
sf.write(output_file_path, combined_sound, sr)
print(f"Combined drum sound saved to '{output_file_path}'")