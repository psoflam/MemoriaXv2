# Import the init_db function from the storage.database module
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.storage.database import init_db, store_session_memory, summarize_session, log_session_message
# Import the process_input function from the core.chatbot module
from memoriax2.core.chatbot import process_input, summarize_session
import datetime
import uuid
from memoriax2.memory.index_engine import get_memory_index

# Generate a random session_id at startup
session_id = str(uuid.uuid4())

def main():
    # Initialize the database connection
    conn = init_db()

    # Initialize the shared MemoryIndex instance
    memory_index = get_memory_index(384)  # Assuming 384 is the embedding dimension
    memory_index.load_index_from_db(conn)  # Load data from the database

    while True:
        # Get user input from the console
        user_input = input("You: ")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] You: {user_input}")

        # Process the user input and get the response from the chatbot
        response, emotion = process_input(user_input, conn, session_id, memory_index)
        print(f"[{timestamp}] Bot: {response}")

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
