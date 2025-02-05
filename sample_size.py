import pickle

# Load the sample lengths
sample_lengths_path = 'combined_features/sample_lengths.pkl'
with open(sample_lengths_path, 'rb') as f:
    sample_lengths = pickle.load(f)

# Print the contents of sample_lengths
print(f"Sample lengths: {sample_lengths}")
print(f"Number of sample lengths: {len(sample_lengths)}")