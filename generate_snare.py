import numpy as np
import pickle
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf

# Load the trained VAE model
vae = load_model('vae_combined_model.keras', compile=False)

# Load the scaler and PCA
with open('combined_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('combined_pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Function to generate new combined features
def generate_combined_features(num_samples=1):
    latent_dim = 10  # Dimension of the latent space
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    generated_features = vae.predict(z_sample)
    return generated_features

# Generate new combined features
new_combined_features = generate_combined_features(num_samples=1)

# Inverse transform the features using PCA and scaler
new_combined_features = pca.inverse_transform(new_combined_features)
new_combined_features = scaler.inverse_transform(new_combined_features)

# Function to convert features to audio (example using inverse MFCC)
def features_to_audio(features):
    # This is a placeholder function. You need to implement the actual conversion
    # from features to audio based on your feature extraction method.
    # For example, if you used MFCCs, you need to use inverse MFCC to get the audio.
    audio = np.zeros(44100)  # Placeholder for 1 second of audio at 44100 Hz
    return audio

# Convert the generated features to audio
new_combined_audio = features_to_audio(new_combined_features[0])

# Save the generated combined sound to a new file
output_file_path = 'generated_combined_sound.wav'
sf.write(output_file_path, new_combined_audio, 44100)
print(f"Generated combined sound saved to '{output_file_path}'")