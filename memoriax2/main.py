import ctypes
import builtins
import os
import sys

safe_print = builtins.print  # backup real print

# Ensure sys is imported before use
def suppress_stderr_windows():
    # Redirect Windows stderr (only affects C/C++ libs like llama.cpp)
    stderr_fileno = sys.stderr.fileno()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), stderr_fileno)

# Ensure os is imported before use
if os.name == 'nt':
    suppress_stderr_windows()

# Import the init_db function from the storage.database module
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.storage.database import init_db, store_session_memory, summarize_session, log_session_message
# Import the process_input function from the core.chatbot module
from memoriax2.core.chatbot import process_input, summarize_session
import datetime
import uuid
from memoriax2.memory.index_engine import get_memory_index

from memoriax2.utils.log import silence_prints, debug_print
# silence_prints()

# Generate a random session_id at startup
session_id = str(uuid.uuid4())

def main():
    # Redirect stdout to suppress loading messages
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Initialize the database connection
    conn = init_db()

    # Initialize the shared MemoryIndex instance
    memory_index = get_memory_index(384)  # Assuming 384 is the embedding dimension
    memory_index.load_index_from_db(conn)  # Load data from the database

    # Restore stdout
    sys.stdout.close()
    sys.stdout = original_stdout

    while True:
        # Get user input from the console
        user_input = input("You: ").strip()

        # Debug logging
        print(f"[DEBUG] Received input: {repr(user_input)}")

        # Check for multiple lines
        if "\n" in user_input:
            print("[WARNING] Multiline input detected. Ignoring for safety.")
            continue

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_print(f"[{timestamp}] You: {user_input}")

        # Process the user input and get the response from the chatbot
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
    exit_session(conn)

# At exit, summarize the session and allow user to approve memories to retain
def exit_session(conn):
    summarize_session(conn, session_id)

if __name__ == "__main__":
    # Call the main function to start the program
    main()

# Example of using debug_print
# debug_print("Loaded FAISS index")
