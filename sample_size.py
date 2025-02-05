import pickle

# Load the sample sizes
sample_sizes_path = 'combined_features/sample_sizes.pkl'
with open(sample_sizes_path, 'rb') as f:
    sample_sizes = pickle.load(f)

# Print the contents of sample_sizes
print(f"Sample sizes: {sample_sizes}")
print(f"Number of sample sizes: {len(sample_sizes)}")