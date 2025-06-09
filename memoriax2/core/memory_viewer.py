import sqlite3
from fact_memory_engine import init_fact_db, extract_user_facts, respond_with_fact_if_available
from fact_memory import store_fact, lookup_fact, list_all_facts, extract_name

# Connect to the database
conn = sqlite3.connect('memory.db')
cursor = conn.cursor()

# Function to dump all memory entries
def dump_memory_entries():
    """Fetch and print all memory entries from the database."""
    cursor.execute("SELECT value, memory_type, emotion FROM memory")
    memories = cursor.fetchall()
    if not memories:
        print("No memories found.")
        return

    print("Memory Entries:")
    for value, mem_type, emotion in memories:
        print(f"- Value: {value}, Type: {mem_type}, Emotion: {emotion}")

# Initialize the fact database
init_fact_db()

# Example usage of extracting user facts and responding with facts
# This is a placeholder for where you would integrate these calls in your main logic
user_input = ""  # Replace with actual user input
retrieved_memories = []  # Replace with actual retrieved memories
extract_user_facts(user_input)
fallback_response = respond_with_fact_if_available(user_input, retrieved_memories)
if fallback_response:
    print(fallback_response)

# Example integration of fact memory layer
# Replace with actual user input retrieval logic
def process_user_input(user_input):
    # Example usage of storing a fact
    if "my name is" in user_input.lower() or "i am " in user_input.lower():
        extracted_name = extract_name(user_input)  # Use the regex-based name extractor
        store_fact("user_name", extracted_name)

    # Example usage of looking up a fact
    if "your name" in user_input.lower():
        user_name = lookup_fact("user_name")
        if user_name:
            print(f"Yes, of course! Your name is {user_name}, isn't it?")
        else:
            print("I'm not sure I remember your name yet. Could you tell me again?")

# Example call to process user input
# This should be replaced with the actual input loop or event handler
user_input = ""  # Replace with actual user input
process_user_input(user_input)

# Run the memory dump function if this script is executed
if __name__ == "__main__":
    dump_memory_entries()

# Close the database connection
conn.close() 