from datasets import load_dataset
import random
import json

# Load the datasets
print("Loading datasets...")
daily_dialog = load_dataset('daily_dialog')
# Assuming OpenSubtitles is pre-filtered and available locally
open_subtitles = load_dataset('open_subtitles', split='train[:1%]')  # Load a small subset for demonstration

# Define emotion tags
emotion_tags = ["neutral", "sad", "happy", "angry", "reflective"]

# Function to preprocess a dialog entry
def preprocess_entry(entry):
    # Example preprocessing logic
    input_text = entry['text'][:100]  # Truncate to 100 tokens
    emotion = random.choice(emotion_tags)  # Randomly assign an emotion
    context = "You said: 'This is a fake memory recall.' [0.75, fact]"
    return {
        "input": input_text,
        "context": [context],
        "emotion": emotion,
        "type": "fact",
        "response": "This is a placeholder response."
    }

# Preprocess the datasets
def preprocess_datasets():
    processed_data = []
    for entry in daily_dialog['train']:
        processed_data.append(preprocess_entry(entry))
    for entry in open_subtitles:
        processed_data.append(preprocess_entry(entry))
    return processed_data

# Save the processed data to a JSON file
def save_to_json(data, filename="data/processed_datasets.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# Main execution
def main():
    processed_data = preprocess_datasets()
    save_to_json(processed_data)

if __name__ == "__main__":
    main() 