import uuid
from datetime import datetime

from memoriax2.core.chatbot import process_input, summarize_session
from memoriax2.memory.index_engine import MemoryIndex
from memoriax2.storage.database import init_db, log_session_message


def run_terminal() -> None:
    """Simple command line interface for chatting with MemoriaX."""
    conn = init_db()
    memory_index = MemoryIndex(384)
    memory_index.load_index_from_db(conn)
    session_id = str(uuid.uuid4())

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            response, emotion = process_input(user_input, conn, session_id, memory_index)
            print(f"[{timestamp}] Bot: {response}")
            log_session_message(conn, session_id, user_input, response, emotion)
    finally:
        summarize_session(conn, session_id)


if __name__ == "__main__":
    run_terminal()
