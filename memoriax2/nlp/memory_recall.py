from sentence_transformers import SentenceTransformer, util
import numpy as np
import sqlite3
from memoriax2.memory.index_engine import MemoryIndex
from memoriax2.nlp.embedding import embed_text
from memoriax2.nlp.emotion import detect_emotion

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_dim():
    sample = embed_text("sample text")
    return len(sample)

# Initialize MemoryIndex with the correct embedding dimension at runtime
embedding_dim = get_embedding_dim()
memory_index = MemoryIndex(embedding_dim)

def store_embedding(conn, key, embedding):
    """Store the embedding in the database."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO memory_embeddings (key, embedding) VALUES (?, ?)", (key, embedding.tobytes()))
    conn.commit()

def retrieve_similar_memories(input_text, conn, top_k=3, recent_memory_limit=5):
    try:
        input_embedding = embed_text(input_text)
        # Query MemoryIndex instead of calculating cosine similarities manually
        top_memory_ids = memory_index.query_similar(input_embedding, top_k)

        # Fetch memory texts and keys from the database using the retrieved IDs
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.key, sm.user_input 
            FROM memory_embeddings m 
            JOIN session_messages sm ON m.key = sm.user_input 
            WHERE m.key IN ({})
        """.format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
        top_memories = cursor.fetchall()

        # Prioritize memories with matching emotional tone
        input_emotion = detect_emotion(input_text)
        prioritized_memories = [
            mem for mem in top_memories 
            if detect_emotion(mem[1]) == input_emotion and mem[0].strip().lower() != "exit"
        ]

        # Limit repetition: Exclude recently used memories
        cursor.execute("SELECT key FROM recent_memories ORDER BY timestamp DESC LIMIT ?", (recent_memory_limit,))
        rows = cursor.fetchall()
        recent_memories = {row[0] for row in rows} if rows else set()
        final_memories = [mem for mem in prioritized_memories if mem[0] not in recent_memories]

        return final_memories
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