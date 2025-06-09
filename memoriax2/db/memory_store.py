import uuid
import numpy as np
from memoriax2.nlp.memory_recall import embed_text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- MEMORY FUNCTIONS --------------------
def store_memory(conn, session_id, user_input, value, memory_index=None, memory_type='fact'):
    cursor = conn.cursor()
    memory_key = f"entry_{uuid.uuid4()}"
    new_embedding = embed_text(value)

    logging.info(f"Storing memory with key: {memory_key}")

    cursor.execute("SELECT key, embedding FROM memory_embeddings ORDER BY rowid DESC LIMIT 20")
    for recent_key, emb_blob in cursor.fetchall():
        existing = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(new_embedding, existing) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing))
        logging.info(f"Similarity with '{recent_key}': {sim}")
        if sim > 0.85:  # Adjusted threshold
            logging.info(f"[SKIP] '{memory_key}' similar to '{recent_key}'")
            return

    cursor.execute("INSERT OR REPLACE INTO memory (key, value, memory_type) VALUES (?, ?, ?)", (memory_key, value, memory_type))
    cursor.execute("INSERT OR REPLACE INTO memory_embeddings (key, embedding, source_text) VALUES (?, ?, ?)", (memory_key, new_embedding.tobytes(), user_input))
    cursor.execute("INSERT INTO session_memories (session_id, key) VALUES (?, ?)", (session_id, memory_key))

    if memory_index:
        memory_index.add_memory(memory_key, new_embedding)

    logging.info(f"[INFO] Stored memory: {memory_key}")
    conn.commit()
    return memory_key


def retrieve_memory(conn, key):
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memory WHERE key=?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None

if __name__ == "__main__":
    from memoriax2.db.init import init_db
    conn = init_db()
    key = store_memory(conn, "test_session", "What is AI?", "AI is artificial intelligence.", memory_index=None)
    print("Stored key:", key) 