import os
import numpy as np
import pickle
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
from tensorflow.keras.losses import mse

# Define file paths
model_path = os.path.join('saved_combined_model', 'vae_combined_model.keras')
scaler_path = os.path.join('combined_features', 'combined_scaler.pkl')
pca_path = os.path.join('combined_features', 'combined_pca.pkl')
output_file = 'generated_combined_sound.wav'

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

# Load the scaler and PCA
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(pca_path, 'rb') as f:
    pca = pickle.load(f)

# Function to generate new combined features
def generate_combined_features(num_samples=1):
    latent_dim = 20  # Dimension of the latent space, matching the input shape of the VAE model
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    generated_features = vae.predict(z_sample)
    return generated_features

# Generate new combined features
new_combined_features = generate_combined_features(num_samples=1)

# Inverse transform the features using PCA and scaler
new_combined_features = pca.inverse_transform(new_combined_features)
new_combined_features = scaler.inverse_transform(new_combined_features)

# Save the generated audio to a file
sf.write(output_file, new_combined_features, 44100)
print(f"Generated audio saved to {output_file}")