import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))
    
    # Output layer for classification
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Load features and labels
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
    print("Features loaded:", features.shape)

with open('features/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    print("Labels loaded:", labels.shape)

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Apply PCA to reduce the number of features
pca = PCA(n_components=20)  # Adjust the number of components as needed
features = pca.fit_transform(features)
print("Features after PCA:", features.shape)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Determine the number of classes
num_classes = labels_categorical.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# Create and compile the model
input_shape = (features.shape[1],)
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('saved_model/drum_model.keras')
print("Model saved to 'saved_model/drum_model.keras'")