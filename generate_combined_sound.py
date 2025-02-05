import os
import numpy as np
import pickle
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
import json

# Define file paths
model_path = os.path.join('saved_combined_model', 'vae_combined_model.keras')
scaler_path = os.path.join('combined_features', 'combined_scaler.pkl')
pca_path = os.path.join('combined_features', 'combined_pca.pkl')
sample_sizes_path = os.path.join('combined_features', 'sample_sizes.pkl')
labels_path = os.path.join('features', 'labels.pkl')
labels_config_path = 'labels_config.json'
output_dir = 'outputted_sounds'
output_file = os.path.join(output_dir, 'generated_combined_sound.wav')

# Register the sampling function
@register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Custom layer for KL divergence loss
@register_keras_serializable()
class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return inputs

# Custom layer for reconstruction loss
@register_keras_serializable()
class ReconstructionLossLayer(layers.Layer):
    def call(self, inputs):
        original, reconstructed = inputs
        reconstruction_loss = mse(original, reconstructed)
        reconstruction_loss *= original.shape[1]
        self.add_loss(tf.reduce_mean(reconstruction_loss))
        return reconstructed

# Load the trained VAE model
vae = load_model(model_path, compile=False)

# Load the scaler, PCA, sample sizes, and labels
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(pca_path, 'rb') as f:
    pca = pickle.load(f)
with open(sample_sizes_path, 'rb') as f:
    sample_sizes = pickle.load(f)
    print(f"Sample sizes loaded: {len(sample_sizes)}")
with open(labels_path, 'rb') as f:
    labels = pickle.load(f)
    print(f"Labels loaded: {len(labels)}")

# Load labels configuration
with open(labels_config_path, 'r') as f:
    labels_config = json.load(f)
    print(f"Labels configuration loaded: {labels_config}")

# Function to generate new combined features
def generate_combined_features(num_samples=100, latent_dim=20):
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    generated_features = vae.predict(z_sample)
    return generated_features

# Compute the number of samples to generate based on the labels configuration
def compute_num_samples(labels_config, labels):
    num_samples = 0
    for label, percentage in labels_config.items():
        indices = [i for i, lbl in enumerate(labels) if any(label in str(l).lower() for l in lbl)]
        print(f"Label: {label}, Indices: {indices}")  # Debug print
        if not indices:
            print(f"No indices found for label: {label}")
            continue
        num_samples += len(indices) * (percentage / 100.0)
    return int(num_samples)

# Compute the number of samples to generate
num_samples = compute_num_samples(labels_config, labels)

# Generate new combined features based on the number of samples
new_combined_features = generate_combined_features(num_samples=num_samples)

# Inverse transform the features using PCA and scaler
new_combined_features = pca.inverse_transform(new_combined_features)
new_combined_features = scaler.inverse_transform(new_combined_features)

# Reshape the features to a 1D array for audio generation
audio_data = new_combined_features.flatten()

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check the shape and content of the generated features
print(f"Generated features shape: {new_combined_features.shape}")
print(f"Generated features: {new_combined_features}")

# Save the generated audio to a file
sf.write(output_file, audio_data, 44100)
print(f"Generated audio saved to {output_file}")