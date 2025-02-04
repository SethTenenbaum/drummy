# Drummy

Drummy is a research project that uses TensorFlow to generate new drum sounds based on existing samples and user-controlled parameters. For example, take 80% of snare samples and combine them with 20% of high hat samples and generate a new drum hit. Another example is to take 808 bass drums and boom bap samples to generate a new 808 boom bap sound, creating sounds from labeled features. The project consists of three main components: feature extraction, neural network training, and sound synthesis.

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

2. **Create Samples Directory**:
   - Create a `samples` directory with subdirectories for different drum sounds. For example:
     ```
     samples/
     ├── kick/
     │   ├── 909/
     │   │   └── 909.wav
     │   └── Croup/
     │       └── 909.wav
     ├── snare/
     │   ├── Acoustic/
     │   │   └── snare.wav
     │   └── Electronic/
     │       └── snare.wav
     └── ...
     ```
   - You can find sample dumps on Google Drive or other sources by searching for them online.

3. **Extract Features**:
   - Run the `process_drums.py` script to extract features from drum samples and save them to files:
     ```sh
     python process_drums.py
     ```

4. **Train the Neural Network**:
   - Run the `train_model.py` script to train a neural network using the extracted features:
     ```sh
     python train_model.py
     ```

5. **Prepare Combined Features**:
   - Run the `prepare_combined_data.py` script to prepare combined features from specified drum labels:
     ```sh
     python prepare_combined_data.py
     ```

6. **Train the VAE**:
   - Run the `train_combined_vae.py` script to train a Variational Autoencoder (VAE) using the combined features:
     ```sh
     python train_combined_vae.py
     ```

7. **Generate New Drum Sounds**:
   - Run the `generate_combined_sound.py` script to generate new drum sounds based on the combined features using the trained VAE:
     ```sh
     python generate_combined_sound.py
     ```

## Example

To generate a new drum sound by combining 80% snare and 20% cymbal:

1. **Prepare Combined Features**:
   ```sh
   python prepare_combined_data.py