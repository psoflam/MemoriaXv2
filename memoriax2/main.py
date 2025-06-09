# Removed the try/except block around imports
# try:
#     print("[DEBUG] Top of file reached")
#     # imports...
# except Exception as e:
#     print(f"[IMPORT ERROR] {e}")

print("[DEBUG] File loaded")

import os
import sys
import builtins
import uuid
import ctypes
import datetime
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.db.init import init_db
from memoriax2.db.session_logger import log_session_message, store_session_memory
from memoriax2.db.confirm_utils import summarize_session
from memoriax2.core.chatbot import process_input, summarize_session
from memoriax2.memory.index_engine import get_memory_index

def safe_print(*args, **kwargs):
    print(*args, **kwargs)

# Generate a random session_id at startup
session_id = str(uuid.uuid4())

print("[DEBUG] File loaded")

def main():
    print("[DEBUG] main() called")

    try:
        # Add debug prints to help identify where the script might be failing
        print("[DEBUG] Starting main function")

        # Initialize the database connection
        print("[DEBUG] Initializing database connection")
        conn = init_db()

        # Initialize the shared MemoryIndex instance
        print("[DEBUG] Initializing MemoryIndex")
        memory_index = get_memory_index(384)  # Assuming 384 is the embedding dimension
        memory_index.load_index_from_db(conn)  # Load data from the database

        print("[DEBUG] Entering main loop")

        while True:
            # Get user input from the console
            user_input = input(">> ").strip()
            if not user_input:
                continue

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            safe_print(f"[{timestamp}] You: {user_input}")

            # Process the user input and get the response from the chatbot
            print("[DEBUG] Processing input")
            response, emotion = process_input(user_input, conn, session_id, memory_index)
            safe_print(f"[{timestamp}] Bot: {response}")

            # Store each response in the current session
            # store_session_memory(conn, session_id, user_input)  # Removed as redundant

            # After each turn, detect emotion and store input, response, emotion
            # process_turn(conn, user_input, response)  # Removed as redundant
            log_session_message(conn, session_id, user_input, response, emotion)

            # End session if user types 'exit'
            if user_input.lower() == 'exit':
                break

        # At exit, summarize the session and allow user to approve memories to retain
        print("[DEBUG] Exiting session")
        exit_session(conn)
    except Exception as e:
        print(f"[FATAL] Exception occurred: {e}")

# At exit, summarize the session and allow user to approve memories to retain
def exit_session(conn):
    print("[DEBUG] Summarizing session")
    summarize_session(conn, session_id)

if __name__ == "__main__":  # ← This gets triggered by -m
    print("[DEBUG] __name__ == '__main__'")
    main()

# Example of using debug_print
# debug_print("Loaded FAISS index")
