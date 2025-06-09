import sqlite3
from fact_memory_engine import init_fact_db, extract_user_facts, respond_with_fact_if_available
from fact_memory import store_fact, lookup_fact, list_all_facts, extract_name
import spacy

# Connect to the database
conn = sqlite3.connect('memory.db')
cursor = conn.cursor()

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to extract name using spaCy's NER
def extract_name(input_text):
    doc = nlp(input_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

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
        store_fact(conn, extracted_name)

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

# Function to retrieve similar memories
# This function is a placeholder and should be integrated with the actual logic for retrieving similar memories
# It retrieves both key and value from the memory database
def retrieve_similar_memories(input_text, conn, memory_index, top_k=3):
    try:
        input_embedding = embed_text(input_text)
        top_memory_ids = memory_index.query_similar(input_embedding, top_k)

        if not top_memory_ids:
            safe_print("No results from FAISS")
            return []

        # Fetch memory texts and keys from the database using the retrieved IDs
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT m.key, mem.value 
            FROM memory_embeddings m 
            JOIN memory mem ON m.key = mem.key 
            WHERE m.key IN ({})
            """.format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
        top_memories = cursor.fetchall()

        # Add detailed logging
        safe_print(f"[DEBUG] Retrieved memory keys: {top_memory_ids}")
        safe_print(f"[DEBUG] Retrieved memory values: {top_memories}")

        return top_memories
    except Exception as e:
        safe_print(f"Error retrieving similar memories: {e}")
        return []

# Update store_fact to use extracted name and store full fact
def store_fact(conn, input_text):
    name = extract_name(input_text)
    if name:
        fact = f"Your name is {name}."
        print(f"[FACT] Stored full name fact: {fact}")
        store_memory(conn, "fact_name", fact)
        memory_index.add_memory("fact_name", embed_text(fact))

# Function to generate a response with memory context
# This function is a placeholder and should be integrated with the actual logic for generating responses
def generate_response_with_memory(user_input, conn):
    similar_memories = retrieve_similar_memories(user_input, conn)
    context = " ".join(similar_memories)
    prompt = f"Conversation (Persona: compassionate, curious):\nUser: {user_input}\nContext: {context}\nRespond accordingly."
    # Placeholder for response generation logic
    print(prompt)  # Replace with actual response generation logic

# Example usage of generating a response with memory context
user_input = "Do you know my name?"  # Replace with actual user input
generate_response_with_memory(user_input, conn)

# Close the database connection
conn.close() 