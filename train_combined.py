import numpy as np
import pickle
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

# Load prepared snare features
with open('snare_features.pkl', 'rb') as f:
    snare_features = pickle.load(f)
print("Snare features loaded:", snare_features.shape)

# Define VAE model
latent_dim = 10  # Dimension of the latent space

# Encoder
inputs = layers.Input(shape=(snare_features.shape[1],))
h = layers.Dense(128, activation='relu')(inputs)
h = layers.Dense(64, activation='relu')(h)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = layers.Dense(64, activation='relu')
decoder_h2 = layers.Dense(128, activation='relu')
decoder_mean = layers.Dense(snare_features.shape[1], activation='sigmoid')
h_decoded = decoder_h(z)
h_decoded = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, x_decoded_mean)

# VAE loss
reconstruction_loss = mse(inputs, x_decoded_mean)
reconstruction_loss *= snare_features.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam())

# Train the VAE
vae.fit(snare_feimport numpy as np
import pickle
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

# Load prepared combined features
with open('combined_features.pkl', 'rb') as f:
    combined_features = pickle.load(f)
print("Combined features loaded:", combined_features.shape)

# Define VAE model
latent_dim = 10  # Dimension of the latent space

# Encoder
inputs = layers.Input(shape=(combined_features.shape[1],))
h = layers.Dense(128, activation='relu')(inputs)
h = layers.Dense(64, activation='relu')(h)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = layers.Dense(64, activation='relu')
decoder_h2 = layers.Dense(128, activation='relu')
decoder_mean = layers.Dense(combined_features.shape[1], activation='sigmoid')
h_decoded = decoder_h(z)
h_decoded = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, x_decoded_mean)

# VAE loss
reconstruction_loss = mse(inputs, x_decoded_mean)
reconstruction_loss *= combined_features.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam())

# Train the VAE
vae.fit(combined_features, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained VAE model
vae.save('vae_combined_model.keras')
print("VAE model saved to 'vae_combined_model.keras'")atures, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained VAE model
vae.save('vae_snare_model.keras')
print("VAE model saved to 'vae_snare_model.keras'")