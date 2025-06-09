import sqlite3

# Path to the database file
DB_PATH = 'memoriax2/data/fact_memory.db'

# SQL to create the memory_embeddings table
CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS memory_embeddings (
    key TEXT PRIMARY KEY,
    embedding BLOB
);
'''

# SQL to create the memory table
CREATE_MEMORY_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT,
    memory_type TEXT
);
'''

# SQL to add the source_text column to memory_embeddings table
ALTER_TABLE_SQL = '''
ALTER TABLE memory_embeddings ADD COLUMN source_text TEXT;
'''

# SQL to create the session_memories table
CREATE_SESSION_MEMORIES_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS session_memories (
    session_id TEXT,
    key TEXT,
    PRIMARY KEY (session_id, key)
);
'''

# SQL to create the recent_memories table
CREATE_RECENT_MEMORIES_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS recent_memories (
    key TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''

def check_and_create_table():
    """Check if the memory_embeddings, memory, session_memories, and recent_memories tables exist and create them if they don't. Also, ensure the memory_embeddings table has the source_text column."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        cursor.execute(CREATE_MEMORY_TABLE_SQL)  # Create memory table
        cursor.execute(CREATE_SESSION_MEMORIES_TABLE_SQL)  # Create session_memories table
        cursor.execute(CREATE_RECENT_MEMORIES_TABLE_SQL)  # Create recent_memories table
        # Check if source_text column exists
        cursor.execute("PRAGMA table_info(memory_embeddings);")
        columns = [column[1] for column in cursor.fetchall()]
        if "source_text" not in columns:
            cursor.execute(ALTER_TABLE_SQL)
        conn.commit()
        print("[INFO] Checked and ensured all necessary tables and columns exist.")
    except sqlite3.Error as e:
        print(f"[ERROR] {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_and_create_table() 