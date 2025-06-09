import numpy as np
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.nlp.memory_recall import embed_text

# -------------------- RECENT MEMORY FUNCTIONS --------------------
def retrieve_similar_memories(input_text, conn, memory_index, top_k=3, recent_memory_limit=5):
    try:
        input_emb = embed_text(input_text)
        if not isinstance(input_emb, np.ndarray):
            print("[ERROR] Invalid embedding")
            return []

        top_ids = memory_index.query_similar(input_emb, top_k)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT m.key, sm.response 
            FROM memory_embeddings m 
            JOIN session_memories smem ON m.key = smem.key
            JOIN session_messages sm ON smem.key = sm.user_input
            WHERE m.key IN ({','.join('?' for _ in top_ids)})
        """, top_ids)
        memories = cursor.fetchall()

        input_emotion = detect_emotion(input_text)
        prioritized = [m for m in memories if detect_emotion(m[1]) == input_emotion and m[0].strip().lower() != "exit"]

        cursor.execute("SELECT key FROM recent_memories ORDER BY timestamp DESC LIMIT ?", (recent_memory_limit,))
        recent = {row[0] for row in cursor.fetchall()}

        return [m for m in prioritized if m[0] not in recent]
    except Exception as e:
        print(f"[ERROR] retrieve_similar_memories: {e}")
        return []


def fetch_recent_memory_context(conn, user_input, memory_index):
    try:
        similar = retrieve_similar_memories(user_input, conn, memory_index)
        return "\n".join(similar) if similar else "No relevant memories found."
    except Exception as e:
        print(f"[ERROR] fetch_recent_memory_context: {e}")
        return "No context available"


def log_recent_memory(conn, key):
    conn.execute("INSERT INTO recent_memories (key) VALUES (?)", (key,))
    conn.commit()


if __name__ == "__main__":
    from memoriax2.db.init import init_db
    conn = init_db()
    log_recent_memory(conn, "entry_8d2845b2-45a8-418c-9c3a-5e15281a150f")
    print("Recent memory logged.") 