from sentence_transformers import SentenceTransformer, util
import numpy as np
import sqlite3
from memoriax2.memory.index_engine import MemoryIndex
from memoriax2.nlp.embedding import embed_text

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

def retrieve_similar_memories(input_text, conn, top_k=3):
    """Retrieve top K similar memories based on the input text using MemoryIndex."""
    input_embedding = embed_text(input_text)
    # Query MemoryIndex instead of calculating cosine similarities manually
    top_memory_ids = memory_index.query_similar(input_embedding, top_k)
    
    # Fetch memory texts from the database using the retrieved IDs
    cursor = conn.cursor()
    cursor.execute("SELECT key FROM memory_embeddings WHERE key IN ({})".format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
    top_memories = [row[0] for row in cursor.fetchall()]
    return top_memories 