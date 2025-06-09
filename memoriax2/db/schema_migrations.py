# -------------------- SCHEMA MIGRATIONS --------------------
def ensure_schema(conn):
    cursor = conn.cursor()
    column_checks = {
        "memory": "memory_type",
        "memory_embeddings": "source_text"
    }
    
    for table, column in column_checks.items():
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    conn.commit()


def update_memory_embeddings_schema(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(memory_embeddings)")
    cols = [c[1] for c in cursor.fetchall()]
    if "source_text" not in cols:
        cursor.execute("ALTER TABLE memory_embeddings ADD COLUMN source_text TEXT")
    conn.commit() 