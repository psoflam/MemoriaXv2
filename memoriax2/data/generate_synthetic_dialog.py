import random
import json
import os

# Define the persona traits, memory types, and emotion tags
persona_traits = ["empathetic", "supportive", "insightful", "reflective"]
memory_types = ["fact", "feeling", "goal", "reflection"]
emotion_tags = ["neutral", "sad", "happy", "angry", "reflective"]

# Function to generate a synthetic dialog entry
def generate_entry():
    input_text = "I'm scared I'll fail my goals."
    context = [
        "You said: 'I want to move to Japan someday.' [0.89, goal]",
        "You said: 'I'm overwhelmed at work lately.' [0.84, feeling]"
    ]
    emotion = random.choice(emotion_tags)
    response = "That's okay to feel that way. I'm proud of how far you've come."
    return {
        "input": input_text,
        "context": context,
        "emotion": emotion,
        "type": "feeling",
        "response": response
    }

# Generate synthetic dialog data
def generate_synthetic_data(num_examples_per_emotion=1000):
    data = []
    for _ in range(num_examples_per_emotion):
        for emotion in emotion_tags:
            entry = generate_entry()
            entry["emotion"] = emotion
            data.append(entry)
    random.shuffle(data)
    return data

# Ensure the data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Save the generated data to a JSON file
def save_to_json(data, filename="synthetic_data.json"):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# Main execution
def main():
    synthetic_data = generate_synthetic_data()
    save_to_json(synthetic_data)

if __name__ == "__main__":
    main() 