import os
from dotenv import load_dotenv
from memoriax2.memory.index_engine import MemoryIndex
from memoriax2.nlp.memory_recall import embed_text

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

    conn.commit()
    return conn

def store_memory(conn, key, value):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    
    # Embed the value and add to MemoryIndex
    vector = embed_text(value)
    memory_index.add_memory(key, vector)

def retrieve_memory(conn, key):
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memory WHERE key=?", (key,))
    result = cursor.fetchone()
    return result[0] if result else None

def store_in_db(conn, user_input, response, emotion):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (user_input, response, emotion)
        VALUES (?, ?, ?)
    """, (user_input, response, emotion))
    conn.commit()

def store_session_memory(conn, session_id, key):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO session_memories (session_id, key) VALUES (?, ?)", (session_id, key))
    conn.commit()

def mark_confirmed(conn, session_id, key):
    cursor = conn.cursor()
    cursor.execute("UPDATE session_memories SET confirmed = 1 WHERE session_id = ? AND key = ?", (session_id, key))
    conn.commit()