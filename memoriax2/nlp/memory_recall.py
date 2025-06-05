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

def retrieve_similar_memories(input_text, conn, memory_index, top_k=3, recent_memory_limit=5, requested_type=None):
    try:
        # Embed the input text
        input_embedding = embed_text(input_text)
        
        # Add a type check for the embedding
        if not isinstance(input_embedding, np.ndarray):
            print("[EMBEDDING ERROR] Failed to embed input_text:", input_text)
            return []

        # Log the embedding shape
        print("Embedding shape:", input_embedding.shape)
        print("Index length:", len(memory_index))

        # Use input_embedding instead of input_text
        top_memory_ids = memory_index.query_similar(input_embedding, top_k)

        # Debug print for the returned key list
        print("Returned key list from query_similar:", top_memory_ids)

        if not top_memory_ids:
            print("No results from FAISS")
            return []

        # Fetch memory texts and keys from the database using the retrieved IDs
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.key, mem.value, mem.memory_type
            FROM memory_embeddings m 
            JOIN memory mem ON m.key = mem.key 
            WHERE m.key IN ({})
        """.format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
        top_memories = cursor.fetchall()

        # Log the retrieved memories
        print("Retrieved memories:", top_memories)

        # Filter by requested type if specified
        if requested_type:
            top_memories = [mem for mem in top_memories if mem[2] == requested_type]

        # Prioritize memories with matching emotional tone
        input_emotion = detect_emotion(input_text)
        prioritized_memories = [
            mem for mem in top_memories 
            if detect_emotion(mem[1]) == input_emotion and mem[0].strip().lower() != "exit"
        ]

        # Log prioritized memories
        print("Prioritized memories:", prioritized_memories)

        # Limit repetition: Exclude recently used memories
        cursor.execute("SELECT key FROM recent_memories ORDER BY timestamp DESC LIMIT ?", (recent_memory_limit,))
        rows = cursor.fetchall()
        recent_memories = {row[0] for row in rows} if rows else set()
        final_memories = [mem for mem in prioritized_memories if mem[0] not in recent_memories]

        # Log final memories
        print("Final memories after excluding recent ones:", final_memories)

        # Remove duplicate entries
        final_memories = list(set(final_memories))

        # Calculate similarity scores and append to memory context lines
        final_memories_with_scores = []
        for mem in final_memories:
            key, text, _ = mem
            embedding = embed_text(text)
            score = np.dot(input_embedding, embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(embedding))
            final_memories_with_scores.append((key, text, score))

        # Log each matched memory and its score
        for key, text, score in final_memories_with_scores:
            print(f"Matched memory: {text} (score: {score})")

        # Add confidence tags
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