def mark_confirmed(conn, session_id, key):
    """Mark a session memory as confirmed in the database."""
    cursor = conn.cursor()
    cursor.execute("UPDATE session_memories SET confirmed = 1 WHERE session_id = ? AND key = ?", (session_id, key))
    conn.commit() 