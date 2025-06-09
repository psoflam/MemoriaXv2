import json
from memoriax2.nlp.memory_recall import embed_text, store_embedding
from memoriax2.db.init import init_db

# Initialize the database connection
conn = init_db()

# Simulate a short conversation log
conversation = [
    {"user": "Hello, how are you?"},
    {"bot": "I'm just a bot, but I'm here to help!"},
    {"user": "Can you tell me a joke?"},
    {"bot": "Why did the scarecrow win an award? Because he was outstanding in his field!"},
    {"user": "That's funny!"}
]

# Embed user lines and store them in memory
for i, line in enumerate(conversation):
    if "user" in line:
        text = line["user"]
        embedding = embed_text(text)
        key = f"user_line_{i+1}"
        store_embedding(conn, key, embedding)
        line["key"] = key

# Save the conversation text and memory keys to a JSON file
with open('train.json', 'w') as f:
    json.dump(conversation, f, indent=2)

print("Conversation log and memory keys saved to train.json") 