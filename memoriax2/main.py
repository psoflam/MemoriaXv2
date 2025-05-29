# Import the init_db function from the storage.database module
from memoriax2.storage.database import init_db
# Import the process_input function from the core.chatbot module
from memoriax2.core.chatbot import process_input

def main():
    # Initialize the database connection
    conn = init_db()

    while True:
        # Get user input from the console
        user_input = input("You: ")
        # Process the user input and get the response from the chatbot
        response = process_input(user_input, conn)
        # Print the chatbot's response
        print(f"Bot: {response}")

if __name__ == "__main__":
    # Call the main function to start the program
    main()
