import sqlite3

# Function to create a database connection

def create_connection(db_file):
    """ create a database connection to the SQLite database specified by db_file """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"[INFO] Connected to database: {db_file}")
    except sqlite3.Error as e:
        print(f"[ERROR] {e}")
    return conn

def store_in_db(conn, session_id, memory_key, response, emotion):
    """Store a session memory and message in the database."""
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