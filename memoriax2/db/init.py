import os
import logging
try:
    from pysqlcipher3 import dbapi2 as sqlite
    USE_ENCRYPTION = True
except ImportError:
    import sqlite3 as sqlite
    USE_ENCRYPTION = False
from dotenv import load_dotenv
from memoriax2.db.schema_migrations import ensure_schema

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- DATABASE INIT --------------------
def init_db():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, '..', 'memory.db')
    logging.info(f"Initializing database at: {db_path}")
    conn = sqlite.connect(os.path.normpath(db_path))
    cursor = conn.cursor()

    if USE_ENCRYPTION:
        key = os.getenv('MEMORY_DB_KEY', 'default_dev_key')
        cursor.execute(f"PRAGMA key='{key}'")

    schema_statements = [
        '''CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            memory_type TEXT DEFAULT 'fact')''',
        '''CREATE TABLE IF NOT EXISTS messages (
            user_input TEXT,
            response TEXT,
            emotion TEXT)''',
        '''CREATE TABLE IF NOT EXISTS session_memories (
            session_id TEXT,
            key TEXT,
            confirmed INTEGER DEFAULT 0)''',
        '''CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_input TEXT,
            response TEXT,
            emotion TEXT)''',
        '''CREATE TABLE IF NOT EXISTS memory_embeddings (
            key TEXT PRIMARY KEY,
            embedding BLOB,
            source_text TEXT)''',
        '''CREATE TABLE IF NOT EXISTS recent_memories (
            key TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)'''
    ]

    for stmt in schema_statements:
        cursor.execute(stmt)

    conn.commit()
    ensure_schema(conn)
    logging.info("Database initialized and schema ensured.")
    return conn

# Ensure schema function will be moved to schema_migrations.py 