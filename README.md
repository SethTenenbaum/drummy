# Drummy

Drummy is a research project that uses TensorFlow to generate new drum sounds based on existing samples and user-controlled parameters. For example, take 80% of snare samples and combine them with 20% of high hat samples and generate a new drum hit. The project consists of three main components: feature extraction, neural network training, and sound synthesis.

## Project Structure

- `process_drums.py`: Script for extracting features from drum samples and saving them to files.
- `train_model.py`: Script for training a neural network using the extracted features.
- `prepare_combined_data.py`: Script for preparing combined features from specified drum labels.
- `train_combined_vae.py`: Script for training a Variational Autoencoder (VAE) using the combined features.
- `generate_combined_sound.py`: Script for generating new drum sounds based on the combined features using the trained VAE.

## Setup

1. **Install Dependencies**:
   - Ensure you have Python 3.9 or later installed.
   - Install the required Python packages using pip:
     ```sh
     pip install -r requirements.txt
     ```

2. **Extract Features**:
   - Run the `process_drums.py` script to extract features from drum samples and save them to files:
     ```sh
     python process_drums.py
     ```

3. **Train the Neural Network**:
   - Run the `train_model.py` script to train a neural network using the extracted features:
     ```sh
     python train_model.py
     ```

4. **Prepare Combined Features**:
   - Run the `prepare_combined_data.py` script to prepare combined features from specified drum labels (e.g., 80% snare and 20% cymbal):
     ```sh
     python prepare_combined_data.py
     ```

5. **Train the VAE**:
   - Run the `train_combined_vae.py` script to train a Variational Autoencoder (VAE) using the combined features:
     ```sh
     python train_combined_vae.py
     ```

6. **Generate New Drum Sounds**:
   - Run the `generate_combined_sound.py` script to generate new drum sounds based on the combined features using the trained VAE:
     ```sh
     python generate_combined_sound.py
     ```

## Usage

1. **Feature Extraction**:
   - The `process_drums.py` script extracts features from drum samples and saves them to files. Ensure your drum samples are organized in the `samples/` directory.

2. **Neural Network Training**:
   - The `train_model.py` script trains a neural network using the extracted features. The trained model is saved to the `saved_model/` directory.

3. **Prepare Combined Features**:
   - The `prepare_combined_data.py` script prepares combined features from specified drum labels (e.g., 80% snare and 20% cymbal). The combined features, scaler, and PCA are saved to files.

4. **Train the VAE**:
   - The `train_combined_vae.py` script trains a Variational Autoencoder (VAE) using the combined features. The trained VAE model is saved to the `saved_model/` directory.

5. **Generate New Drum Sounds**:
   - The `generate_combined_sound.py` script generates new drum sounds based on the combined features using the trained VAE. The generated drum sound is saved to a new file.

## Example

To generate a new drum sound by combining 80% snare and 20% cymbal:

1. **Prepare Combined Features**:
   ```sh
   python prepare_combined_data.py