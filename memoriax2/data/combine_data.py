import json
import random
import os

# Ensure the data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Load synthetic data
def load_synthetic_data(filename="synthetic_data.json"):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'r') as f:
        return json.load(f)

# Load processed external data
def load_processed_data(filename="processed_datasets.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# Combine, shuffle, and validate data
def combine_and_shuffle_data(synthetic_data, processed_data, target_size=5000):
    combined_data = synthetic_data + processed_data
    random.shuffle(combined_data)
    # Ensure the data size is within the target range
    if len(combined_data) > target_size:
        combined_data = combined_data[:target_size]
    return combined_data

# Save the combined data to a JSON file
def save_to_json(data, filename="train.json"):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# Main execution
def main():
    synthetic_data = load_synthetic_data()
    processed_data = load_processed_data()
    combined_data = combine_and_shuffle_data(synthetic_data, processed_data)
    save_to_json(combined_data)

if __name__ == "__main__":
    main() 