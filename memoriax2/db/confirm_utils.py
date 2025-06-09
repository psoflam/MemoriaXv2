# -------------------- CONFIRMATION UTILITIES --------------------
def mark_confirmed(conn, session_id, key):
    conn.execute("UPDATE session_memories SET confirmed = 1 WHERE session_id = ? AND key = ?", (session_id, key))
    conn.commit()


def get_session_summary(conn, session_id):
    cursor = conn.execute('SELECT user_input, response, emotion FROM session_messages WHERE session_id = ?', (session_id,))
    return [{'user_input': row[0], 'response': row[1], 'emotion': row[2]} for row in cursor.fetchall()]


def summarize_session(conn, session_id):
    try:
        print(f"[SESSION] Reviewing potential memories for session {session_id}")
        cursor = conn.cursor()
        cursor.execute("SELECT key FROM session_memories WHERE session_id = ? AND confirmed = 0", (session_id,))
        potentials = cursor.fetchall()

        if not potentials:
            print("[INFO] No unconfirmed memories.")
            return

        seen = set()
        for mem in potentials:
            key = mem[0]
            if key.strip().lower() == "exit" or key in seen:
                continue
            seen.add(key)
            print(f"- {key}")
            if input(f"Remember this? {key} (yes/no): ").strip().lower() == 'yes':
                mark_confirmed(conn, session_id, key)
                val = retrieve_memory(conn, key)
                log_session_message(conn, session_id, key, val, "neutral")

    except Exception as e:
        print(f"[ERROR] summarize_session: {e}")


if __name__ == "__main__":
    from memoriax2.db.init import init_db
    from memoriax2.db.memory_store import store_memory
    from memoriax2.db.recent import log_recent_memory

    conn = init_db()
    session_id = "test_session"

    # Step 1: Store a memory
    key = store_memory(conn, session_id, "What is AI?", "AI is artificial intelligence.", memory_index=None)
    print("Stored key:", key)

    # Step 2: Confirm the memory
    summarize_session(conn, session_id)

    # Step 3: Log recent memory
    log_recent_memory(conn, key)

    # Step 4: Simulate quitting and restarting
    print("Quitting session...")
    conn.close()

    # Restart and check if memory is loaded
    conn = init_db()
    print("Session restarted.")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memory")
    count = cursor.fetchone()[0]
    print(f"Loaded {count} memories.") 