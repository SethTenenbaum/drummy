# Drummy

Drummy is a research project that uses TensorFlow to generate new drum sounds from existing samples based on user-defined parameters. For example, you can create a new drum hit by combining 80% snare samples with 20% high hat samples. Another example is blending 808 bass drums with boom bap samples to produce a unique 808 boom bap sound. The project consists of three main components: feature extraction, neural network training, and sound synthesis.

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
   - Create a `labels_config.json` file to specify the labels and their corresponding percentages. For example:
     ```json
     {
         "snare": 10,
         "boombap": 20,
         "808": 70
     }
     ```
   - Run the `prepare_combined_data.py` script to prepare combined features from specified drum labels using the `labels_config.json` file:
     ```sh
     python prepare_combined_data.py
     ```

6. **Train the VAE**:
   - Run the `train_combined_vae.py` script to train a Variational Autoencoder (VAE) using the combined features:
     ```sh
     python train_combined_vae.py
     ```

7. **Generate New Drum Sounds**:
   - Run the `generate_combined_sound.py` script to generate new drum sounds based on the combined features using the trained VAE.   The generated drum sound will be saved to generated_combined_sound.wav:
     ```sh
     python generate_combined_sound.py
     ```

Requirements
- Python 3.9 or later
- TensorFlow
- Librosa
- Soundfile
- Scikit-learn

