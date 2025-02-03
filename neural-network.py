import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output layer for tuning parameter
    return model

input_shape = (features.shape[1],)
model = create_model(input_shape)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(features, labels, epochs=50, batch_size=8)
model.save('drum_synth_model.h5')