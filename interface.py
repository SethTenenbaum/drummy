import tensorflow as tf
import numpy as np
import soundfile as sf

# Load the trained model
model = tf.keras.models.load_model('drum_synth_model.h5')

def generate_drum_sound(tuning_param):
    # Generate features based on the tuning parameter
    features = np.array([tuning_param] * 13).reshape(1, -1)
    generated_sound = model.predict(features)
    return generated_sound

# Example usage
tuning_param = 0.5  # Adjust this parameter to tune the sound
generated_sound = generate_drum_sound(tuning_param)

# Save the generated sound to a file