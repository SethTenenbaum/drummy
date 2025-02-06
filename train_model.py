import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models, Input, regularizers
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
    
    # Output layer for multi-label classification
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    
    return model

# Load features, labels, sample sizes, and ADSR parameters
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)
    print("Features loaded:", features.shape)

with open('features/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
    print("Labels loaded:", labels.shape)

with open('features/sample_sizes.pkl', 'rb') as f:
    sample_sizes = pickle.load(f)
    sample_sizes = np.array(sample_sizes).reshape(-1, 1)  # Convert to NumPy array and reshape
    print("Sample sizes loaded:", sample_sizes.shape)

with open('features/adsr_params.pkl', 'rb') as f:
    adsr_params = pickle.load(f)
    adsr_params = np.array(adsr_params)  # Convert to NumPy array
    print("ADSR parameters loaded:", adsr_params.shape)

# Concatenate features, sample sizes, and ADSR parameters
features_with_sizes_and_adsr = np.hstack((features, sample_sizes, adsr_params))
print("Features with sample sizes and ADSR parameters shape:", features_with_sizes_and_adsr.shape)

# Normalize features
scaler = StandardScaler()
features_with_sizes_and_adsr = scaler.fit_transform(features_with_sizes_and_adsr)

# Apply PCA to reduce the number of features
pca = PCA(n_components=20)  # Adjust the number of components as needed
features_with_sizes_and_adsr = pca.fit_transform(features_with_sizes_and_adsr)
print("Features after PCA:", features_with_sizes_and_adsr.shape)

# Encode multi-dimensional labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)
print("Labels encoded:", labels_encoded.shape)

# Determine the number of classes
num_classes = labels_encoded.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_with_sizes_and_adsr, labels_encoded, test_size=0.2, random_state=42)

# Create and compile the model
input_shape = (features_with_sizes_and_adsr.shape[1],)
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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