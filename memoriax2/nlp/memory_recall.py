from sentence_transformers import SentenceTransformer, util
import numpy as np
import sqlite3

from memoriax2.utils.generate import generate_with_model
from memoriax2.nlp.embedding import embed_text
from memoriax2.nlp.emotion import detect_emotion

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_dim():
    """Return the dimension of the embedding vector."""
    sample = embed_text("sample text")
    return len(sample)

# Initialize MemoryIndex with the correct embedding dimension at runtime
embedding_dim = get_embedding_dim()

def store_embedding(conn, key, embedding):
    """Store the embedding in the database."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO memory_embeddings (key, embedding) VALUES (?, ?)", (key, embedding.tobytes()))
    conn.commit()

def retrieve_similar_memories(input_text, conn, memory_index, top_k=3, recent_memory_limit=5, requested_type=None):
    """Retrieve memories similar to the input text based on embeddings."""
    try:
        input_embedding = embed_text(input_text)
        if not isinstance(input_embedding, np.ndarray):
            print("[EMBEDDING ERROR] Failed to embed input_text:", input_text)
            return []

        top_memory_ids = memory_index.query_similar(input_embedding, top_k)
        if not top_memory_ids:
            print("No results from FAISS")
            return []

        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.key, mem.value, mem.memory_type
            FROM memory_embeddings m 
            JOIN memory mem ON m.key = mem.key 
            WHERE m.key IN ({})
        """.format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
        top_memories = cursor.fetchall()

        if requested_type:
            top_memories = [mem for mem in top_memories if mem[2] == requested_type]

        input_emotion = detect_emotion(input_text)
        prioritized_memories = [
            mem for mem in top_memories 
            if detect_emotion(mem[1]) == input_emotion and mem[0].strip().lower() != "exit"
        ]

        cursor.execute("SELECT key FROM recent_memories ORDER BY timestamp DESC LIMIT ?", (recent_memory_limit,))
        rows = cursor.fetchall()
        recent_memories = {row[0] for row in rows} if rows else set()
        final_memories = [mem for mem in prioritized_memories if mem[0] not in recent_memories]

        final_memories = list(set(final_memories))

        final_memories_with_scores = []
        for mem in final_memories:
            key, text, _ = mem
            embedding = embed_text(text)
            score = np.dot(input_embedding, embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(embedding))
            final_memories_with_scores.append((key, text, score))

        final_memories_with_scores = [(key, f"{text} [{score:.2f}]") for key, text, score in final_memories_with_scores]

        return final_memories_with_scores
    except Exception as e:
        print(f"Error retrieving similar memories: {e}")
        return []

def populate_memory(conn):
    """Populate memory with emotionally diverse phrases."""
    phrases = [
        "I am so happy today!",
        "I'm feeling very sad right now.",
        "Do you remember our trip to Japan?",
        "The weather is gloomy and it makes me nostalgic.",
        "I am excited about the new project!"
    ]
    
    for i, text in enumerate(phrases):
        key = f"entry_{i+1}"
        embedding = embed_text(text)
        store_embedding(conn, key, embedding) 

def generate_base_response(user_input, context="", emotion="neutral", persona="curious, kind, attentive"):
    """Generate a base response using the model."""
    prompt = f"""<s>[INST] You are MemoriaX, an emotionally intelligent AI companion.

Persona: {persona}
User Emotion: {emotion}
Relevant Memory:
{context if context else "None"}

The user said: "{user_input}"

Now respond as MemoriaX. [/INST]"""
    return generate_with_model(prompt)
