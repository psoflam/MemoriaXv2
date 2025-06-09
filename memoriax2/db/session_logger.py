# -------------------- SESSION LOGGING FUNCTIONS --------------------
def log_session_message(conn, session_id, user_input, response, emotion):
    conn.execute(
        'INSERT INTO session_messages (session_id, user_input, response, emotion) VALUES (?, ?, ?, ?)',
        (session_id, user_input, response, emotion)
    )
    conn.commit()


def store_session_memory(conn, session_id, key):
    conn.execute("INSERT INTO session_memories (session_id, key) VALUES (?, ?)", (session_id, key))
    conn.commit()


if __name__ == "__main__":
    from memoriax2.db.init import init_db
    conn = init_db()
    log_session_message(conn, "test_session", "Hello, how are you?", "I'm fine, thank you.", "neutral")
    print("Session message logged.") 