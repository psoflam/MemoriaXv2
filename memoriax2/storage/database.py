import os
from dotenv import load_dotenv
from memoriax2.memory.index_engine import MemoryIndex
from memoriax2.nlp.memory_recall import embed_text
import numpy as np

# Try to import SQLCipher, otherwise fall back to plain SQLite
try:
    from pysqlcipher3 import dbapi2 as sqlite
    USE_ENCRYPTION = True
except ImportError:
    import sqlite3 as sqlite
    USE_ENCRYPTION = False

# Load environment variables from .env file
load_dotenv()

# Initialize MemoryIndex
memory_index = MemoryIndex()

def init_db():
    conn = sqlite.connect('memory.db')  # Connect to the (possibly encrypted) database
    cursor = conn.cursor()

    # If using SQLCipher, apply encryption key
    if USE_ENCRYPTION:
        key = os.getenv('MEMORY_DB_KEY', 'default_dev_key')
        cursor.execute(f"PRAGMA key='{key}'")

    cursor.execute('''CREATE TABLE IF NOT EXISTS memory (
                        key TEXT PRIMARY KEY,
                        value TEXT
                      )''')
    
    # Optional: table used in store_in_db()
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
                        user_input TEXT,
                        response TEXT,
                        emotion TEXT
                      )''')

    # New table for session memories
    cursor.execute('''CREATE TABLE IF NOT EXISTS session_memories (
                        session_id TEXT,
                        key TEXT,
                        confirmed INTEGER DEFAULT 0
                      )''')

    # Add the session_messages table
    cursor.execute('''CREATE TABLE IF NOT EXISTS session_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        user_input TEXT,
                        response TEXT,
                        emotion TEXT
                      )''')

    conn.commit()
    return conn

def store_memory(conn, key, value):
    cursor = conn.cursor()
    new_embedding = embed_text(value)

    # Get recent memory embeddings
    cursor.execute("SELECT key, embedding FROM memory_embeddings ORDER BY rowid DESC LIMIT 20")
    recent = cursor.fetchall()

    # Compare embeddings
    for recent_key, emb_blob in recent:
        existing = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(new_embedding, existing) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing))
        if sim > 0.9:
            print(f"Skipping memory '{key}' due to similarity with '{recent_key}'")
            return

    cursor.execute("INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (key, value))
    cursor.execute("INSERT OR REPLACE INTO memory_embeddings (key, embedding) VALUES (?, ?)", (key, new_embedding.tobytes()))
    memory_index.add_memory(key, new_embedding)
    conn.commit()

def retrieve_memory(conn, key):
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memory WHERE key=?", (key,))
    result = cursor.fetchone()
    return result[0] if result else None

def store_in_db(conn, session_id, user_input, response, emotion):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO session_memories (session_id, key, confirmed)
        VALUES (?, ?, 0)
    """, (session_id, user_input))
    cursor.execute("""
        INSERT INTO session_messages (session_id, user_input, response, emotion)
        VALUES (?, ?, ?, ?)
    """, (session_id, user_input, response, emotion))
    conn.commit()

def store_session_memory(conn, session_id, key):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO session_memories (session_id, key) VALUES (?, ?)", (session_id, key))
    conn.commit()

def mark_confirmed(conn, session_id, key):
    cursor = conn.cursor()
    cursor.execute("UPDATE session_memories SET confirmed = 1 WHERE session_id = ? AND key = ?", (session_id, key))
    conn.commit()

# Function to log a session message
def log_session_message(conn, session_id, user_input, response, emotion):
    with conn:
        conn.execute(
            'INSERT INTO session_messages (session_id, user_input, response, emotion) VALUES (?, ?, ?, ?)',
            (session_id, user_input, response, emotion)
        )

# Function to get session summary
def get_session_summary(conn, session_id):
    cursor = conn.execute('SELECT user_input, response, emotion FROM session_messages WHERE session_id = ?', (session_id,))
    return [{'user_input': row[0], 'response': row[1], 'emotion': row[2]} for row in cursor.fetchall()]

# Function to summarize session
def summarize_session(conn, session_id):
    try:
        print(f"Session ID: {session_id}")  # Log the session_id
        cursor = conn.cursor()
        cursor.execute("""
            SELECT key FROM session_memories
            WHERE session_id = ? AND confirmed = 0
        """, (session_id,))
        potential_memories = cursor.fetchall()

        print(f"Number of potential memories fetched: {len(potential_memories)}")  # Print number of fetched rows

        if not potential_memories:
            print("No potential memories found.")
            return

        print("Here's what I might remember from today:")
        seen = set()
        for mem in potential_memories:
            if mem[0].strip().lower() == "exit" or mem[0] in seen:
                continue
            seen.add(mem[0])
            print(f"- {mem[0]}")
            user_input = input(f"Should I remember this: {mem[0]}? (yes/no): ")
            if user_input.lower() == 'yes':
                mark_confirmed(conn, session_id, mem[0])
                val = retrieve_memory(conn, mem[0])
                store_in_db(conn, session_id, mem[0], val, "neutral")  # Updated to use store_in_db
    except Exception as e:
        print(f"Error summarizing session: {e}")

def fetch_recent_memory_context(conn, user_input):
    try:
        similar = retrieve_similar_memories(conn, user_input)
        return "\n".join(similar) if similar else "No relevant memories found."
    except Exception as e:
        print(f"Error fetching memory context: {e}")
        return "No context available"

# Update retrieve_similar_memories to pull from session_messages
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
        recent_memories = {row[0] for row in cursor.fetchall()}
        final_memories = [mem for mem in prioritized_memories if mem[0] not in recent_memories]

        return final_memories
    except Exception as e:
        print(f"Error retrieving similar memories: {e}")
        return []