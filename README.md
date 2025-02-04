# Drummy

Drummy is a project that uses TensorFlow to generate new drum sounds based on existing samples and user-controlled parameters. For example, take 80% of snare samples and combine them with 20% of high hat samples and generate a new drum hit.  The project consists of three main components: feature extraction, neural network training, and sound synthesis.

## Project Structure

- `process_drums.py`: Script for extracting features from drum samples and saving them to files.
- `train_model.py`: Script for training a neural network using the extracted features.
- `synth_interface.py`: Script for generating new drum sounds based on user input using the trained model.

## Setup

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Librosa
- Soundfile
- Pickle
- Requests
- BeautifulSoup4

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/drummy.git
    cd drummy
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow numpy librosa soundfile pickle-mixin requests beautifulsoup4
    ```

## Usage

### Step 1: Extract Features

Run the `process_drums.py` script to extract features from your drum samples and save them to files.

### Step 2: Train the Neural Network

Run the `train_model.py` script to train the neural network using the extracted features.

### Step 3: Generate New Drum Sounds

Run the `synth_interface.py` script to generate new drum sounds based on user input using the trained model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.