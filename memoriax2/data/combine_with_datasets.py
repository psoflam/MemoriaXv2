from datasets import load_dataset
import random
import json

# Load the datasets
print("Loading datasets...")
daily_dialog = load_dataset('daily_dialog', trust_remote_code=True)
# Assuming OpenSubtitles is pre-filtered and available locally
open_subtitles = load_dataset('open_subtitles', 'en-hi', split='train[:1%]', trust_remote_code=True)  # Load a small subset for demonstration

# Define emotion tags
emotion_tags = ["neutral", "sad", "happy", "angry", "reflective"]

# Function to generate a response based on emotion
def generate_response(text, emotion):
    if emotion == "sad":
        return "I'm here for you. Want to talk more about it?"
    elif emotion == "happy":
        return "That's wonderful to hear! What made you feel that way?"
    elif emotion == "angry":
        return "I can tell that upset you. What happened?"
    elif emotion == "reflective":
        return "It sounds like you're thinking deeply. Want to explore it together?"
    else:
        return f"I hear you. Tell me more about: {text}"

# Function to preprocess a dialog entry
def preprocess_entry(entry, dataset_name):
    # Determine the input text based on the dataset structure
    if dataset_name == 'DailyDialog':
        input_text = ' '.join(entry['dialog'])[:100]  # Adjust field name as per actual structure
    elif dataset_name == 'OpenSubtitles':
        input_text = entry['translation']['en'][:100]  # Use the English translation
    else:
        input_text = "Unknown dataset structure"
    
    emotion = random.choice(emotion_tags)  # Randomly assign an emotion
    context = f"You said: 'This is a fake memory recall from {dataset_name}.' [0.75, fact]"
    return {
        "input": input_text,
        "context": [context],
        "emotion": emotion,
        "type": "fact",
        "response": generate_response(input_text, emotion)
        
    }

# Preprocess the datasets
def preprocess_datasets():
    processed_data = []
    for entry in daily_dialog['train']:
        processed_data.append(preprocess_entry(entry, 'DailyDialog'))
    for entry in open_subtitles:
        processed_data.append(preprocess_entry(entry, 'OpenSubtitles'))
    return processed_data

# Save the processed data to a JSON file
def save_to_json(data, filename="processed_datasets.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# Main execution
def main():
    processed_data = preprocess_datasets()
    save_to_json(processed_data)

if __name__ == "__main__":
    main() 