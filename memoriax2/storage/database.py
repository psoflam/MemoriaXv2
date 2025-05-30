import os
from dotenv import load_dotenv
from memoriax2.memory.index_engine import MemoryIndex
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.nlp.memory_recall import embed_text
import numpy as np
import uuid  # Add this import at the top of the file

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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, '..', 'memory.db')
    conn = sqlite.connect(os.path.normpath(db_path))  # Connect to the (possibly encrypted) database
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

    # Add the memory_embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            key TEXT PRIMARY KEY,
            embedding BLOB,
            source_text TEXT
        )
    ''')

    # Add the recent_memories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recent_memories (
            key TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    return conn

def store_memory(conn, session_id, user_input, value):
    cursor = conn.cursor()
    memory_key = f"entry_{uuid.uuid4()}"  # Generate a UUID key with prefix 'entry_'
    new_embedding = embed_text(value)

    # Log the mapping of user_input to memory_key
    print(f"Mapping user_input '{user_input}' to memory_key '{memory_key}'")

    # Get recent memory embeddings
    cursor.execute("SELECT key, embedding FROM memory_embeddings ORDER BY rowid DESC LIMIT 20")
    recent = cursor.fetchall()

    # Compare embeddings
    for recent_key, emb_blob in recent:
        existing = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(new_embedding, existing) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing))
        if sim > 0.9:
            print(f"Skipping memory '{memory_key}' due to similarity with '{recent_key}'")
            return

    cursor.execute("INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (memory_key, value))
    cursor.execute("INSERT OR REPLACE INTO memory_embeddings (key, embedding, source_text) VALUES (?, ?, ?)", (memory_key, new_embedding.tobytes(), user_input))  # Add source_text
    cursor.execute("INSERT INTO session_memories (session_id, key) VALUES (?, ?)", (session_id, memory_key))
    memory_index.add_memory(memory_key, new_embedding)
    print("Top FAISS keys now:", memory_index.list_keys())
    # Print the top 3 keys in memory_index for verification
    print("Top 3 keys in memory_index:", memory_index.list_keys()[:3])

    conn.commit()
    return memory_key

def retrieve_memory(conn, key):
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memory WHERE key=?", (key,))
    result = cursor.fetchone()
    return result[0] if result else None

def store_in_db(conn, session_id, memory_key, response, emotion):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO session_memories (session_id, key, confirmed)
        VALUES (?, ?, 0)
    """, (session_id, memory_key))
    cursor.execute("""
        INSERT INTO session_messages (session_id, user_input, response, emotion)
        VALUES (?, ?, ?, ?)
    """, (session_id, memory_key, response, emotion))
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

        # Debug print for the returned key list
        print("Returned key list from query_similar:", top_memory_ids)

        # Fetch memory texts and keys from the database using the retrieved IDs
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.key, sm.response 
            FROM memory_embeddings m 
            JOIN session_memories smem ON m.key = smem.key
            JOIN session_messages sm ON smem.key = sm.user_input
            WHERE m.key IN ({})
        """.format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
        top_memories = cursor.fetchall()

        if not top_memories:
            print("No memories found for the given keys:", top_memory_ids)

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

def log_recent_memory(conn, key):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO recent_memories (key) VALUES (?)", (key,))
    conn.commit()

def store_memory_and_log(conn, session_id, user_input, value, response, emotion):
    memory_key = store_memory(conn, session_id, user_input, value)  # Ensure store_memory returns the memory_key
    store_in_db(conn, session_id, memory_key, response, emotion)
    print(f"Stored memory with key: {memory_key}")

def summarize_session_and_retrieve(conn, session_id):
    summary = summarize_session(conn, session_id)
    for item in summary:
        memory_key = item['key']
        memory_text = retrieve_memory(conn, memory_key)
        print(f"Memory Key: {memory_key}, Text: {memory_text}")

def update_memory_embeddings_schema(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE memory_embeddings ADD COLUMN source_text TEXT")
        conn.commit()
        print("Schema updated successfully.")
    except sqlite.Error as e:
        print(f"An error occurred: {e}")