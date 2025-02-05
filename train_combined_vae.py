import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load combined features
with open('combined_features/combined_features.pkl', 'rb') as f:
    combined_features = pickle.load(f)
print("Combined features loaded:", combined_features.shape)

# Check if the number of features matches the scaler and PCA
expected_num_features = 39  # Update this to match the original number of features

if combined_features.shape[1] != expected_num_features:
    print(f"Warning: Combined features have {combined_features.shape[1]} features, but expected {expected_num_features} features.")
    print("Re-fitting the scaler and PCA on the current combined features.")
    
    # Re-fit the scaler and PCA
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)
    
    pca = PCA(n_components=20)  # Adjust the number of components as needed
    combined_features = pca.fit_transform(combined_features)
    
    # Save the re-fitted scaler and PCA
    with open('combined_features/combined_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('combined_features/combined_pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
else:
    # Load the scaler and PCA
    with open('combined_features/combined_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('combined_features/combined_pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    
    # Apply the scaler and PCA to the combined features
    combined_features = scaler.transform(combined_features)
    combined_features = pca.transform(combined_features)

# Define VAE model
latent_dim = 10  # Dimension of the latent space

# Encoder
inputs = layers.Input(shape=(combined_features.shape[1],))
h = layers.Dense(128, activation='relu')(inputs)
h = layers.Dense(64, activation='relu')(h)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = layers.Dense(64, activation='relu')
decoder_mean = layers.Dense(combined_features.shape[1], activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom layer for KL divergence loss
class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return inputs

# Apply the KL divergence layer
z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])

# Custom layer for reconstruction loss
class ReconstructionLossLayer(layers.Layer):
    def call(self, inputs):
        original, reconstructed = inputs
        reconstruction_loss = mse(original, reconstructed)
        reconstruction_loss *= combined_features.shape[1]
        self.add_loss(tf.reduce_mean(reconstruction_loss))
        return reconstructed

# Apply the reconstruction loss layer
x_decoded_mean = ReconstructionLossLayer()([inputs, x_decoded_mean])

# VAE model
vae = Model(inputs, x_decoded_mean)

# Compile the VAE model
vae.compile(optimizer=Adam())

# Train the VAE
vae.fit(combined_features, combined_features, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained VAE model
vae.save('vae_combined_model.keras')
print("Trained VAE model saved to 'vae_combined_model.keras'")