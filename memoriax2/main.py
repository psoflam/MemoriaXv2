# Import the init_db function from the storage.database module
from memoriax2.storage.database import init_db, store_session_memory, summarize_session
# Import the process_input function from the core.chatbot module
from memoriax2.core.chatbot import process_input
import datetime

def main():
    # Initialize the database connection
    conn = init_db()
    session_id = str(datetime.datetime.now().timestamp())  # Unique session ID based on timestamp

    while True:
        # Get user input from the console
        user_input = input("You: ")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] You: {user_input}")

        # Process the user input and get the response from the chatbot
        response = process_input(user_input, conn)
        print(f"[{timestamp}] Bot: {response}")

        # Store each response in the current session
        store_session_memory(conn, session_id, user_input)

        # End session if user types 'exit'
        if user_input.lower() == 'exit':
            break

    # At the end of the session, print emotional summary and ask for memory confirmation
    summarize_session(conn, session_id)

if __name__ == "__main__":
    # Call the main function to start the program
    main()
