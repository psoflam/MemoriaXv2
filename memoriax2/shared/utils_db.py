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