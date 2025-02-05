import numpy as np
import pickle
import soundfile as sf
from tensorflow.keras.models import load_model

# Load the trained VAE model
vae = load_model('vae_combined_model.keras', compile=False)

# Load the scaler and PCA
with open('combined_features/combined_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('combined_features/combined_pca.pkl', 'rb') as f:
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

# Save the generated audio to a file
output_file = 'generated_combined_sound.wav'
sf.write(output_file, new_combined_features, 44100)
print(f"Generated audio saved to {output_file}")