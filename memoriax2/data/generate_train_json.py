import json

# Sample data
sample_data = [
    {"input": "Hello, how are you?", "response": "I'm good, thank you!", "emotion": "happy"},
    {"input": "I'm feeling sad today.", "response": "I'm sorry to hear that. I'm here for you.", "emotion": "sad"},
    {"input": "What a wonderful day!", "response": "Yes, it's beautiful outside.", "emotion": "joyful"}
]

# Path to save the train.json file
train_json_path = 'memoriax2/data/train.json'

# Write the sample data to train.json
with open(train_json_path, 'w') as f:
    json.dump(sample_data, f, indent=4)

print(f"Sample train.json file created at {train_json_path}") 