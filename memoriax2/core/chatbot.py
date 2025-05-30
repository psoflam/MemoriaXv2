from memoriax2.nlp.processor import analyze_input  # Import the analyze_input function from the memoriax2.nlp.processor module
from memoriax2.storage.database import mark_confirmed, store_in_db, store_memory, retrieve_memory  # Import store_memory and retrieve_memory functions from the memoriax2.storage.database module
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.nlp.memory_recall import retrieve_similar_memories
import faiss
import random

def process_input(user_input, conn, session_id):
    try:
        context = fetch_recent_memory_context(conn, user_input)
        emotion = detect_emotion(user_input)
        response = chat_with_user(user_input, context, emotion)

        # Log the turn
        store_in_db(conn, session_id, user_input, response, emotion)
        return response, emotion
    except Exception as e:
        print(f"Error processing input: {e}")
        return "I'm sorry, something went wrong.", None

def store_memory(user_input, response):
    emotion = detect_emotion(user_input)
    # Store the emotion alongside the user message
    # Assuming there's a function to store data in the database
    store_in_db(user_input, response, emotion)

def generate_response(user_input, conn=None):
    emotion = detect_emotion(user_input)
    context = ""  # placeholder for memory/context logic
    base = generate_base_response(user_input + " " + context)

    if emotion == "sad":
        response = make_response_more_empathetic(base)
    else:
        response = f"Calmly, {base}"

    if conn:
        store_in_db(conn, user_input, response, emotion)

    return response

def generate_base_response(user_input):
    # Basic implementation for generating a base response
    return f"Response to: {user_input}"

def make_response_more_calming(response):
    # Basic implementation to make the response more calming
    return f"Calmly, {response}"

def make_response_more_empathetic(response):
    # Basic implementation to make the response more empathetic
    return f"This is an empathetic response: {response}"

def __init__(self, embedding_dim: int):
    self.index = faiss.IndexFlatL2(embedding_dim)  # Set the index dimension dynamically
    self.id_map = {}

def conversation_mode(user_input, conn):
    # Check for recent messages
    recent_messages = retrieve_memory(conn, 'recent_messages')
    if not recent_messages:
        return "Hello! How can I assist you today?"

    # Recognize check-in phrases
    check_in_phrases = ["how are you", "what's on your mind"]
    if any(phrase in user_input.lower() for phrase in check_in_phrases):
        emotion = detect_emotion(user_input)
        return f"I'm here to listen. How are you feeling today?"

    # Retrieve similar memories with emotional context
    similar_memories = retrieve_similar_memories(user_input, conn)
    context = " ".join(similar_memories)
    response = generate_response(user_input, context)
    return response

def chat_with_user(user_input, context, emotion):
    prompt = f"User: {user_input}\nContext: {context}\nEmotion: {emotion}\nMemoriaX:"
    response = generate_base_response(prompt)
    if emotion == 'sad':
        response = add_empathetic_tone(response)
    return response

def summarize_session(conn, session_id):
    try:
        print(f"Session ID: {session_id}")  # Log the session_id
        cursor = conn.cursor()
        cursor.execute("""
            SELECT key FROM session_memories
            WHERE session_id = ? AND confirmed = 0
        """, (session_id,))
        potential_memories = cursor.fetchall()

        print(f"Number of potential memories fetched: {len(potential_memories)}")  # Print number of fetched rows

        if not potential_memories:
            print("No potential memories found.")
            return

        print("Here's what I might remember from today:")
        for mem in potential_memories:
            print(f"- {mem[0]}")

        for mem in potential_memories:
            user_input = input(f"Should I remember this: {mem[0]}? (yes/no): ")
            if user_input.lower() == 'yes':
                mark_confirmed(conn, session_id, mem[0])
                val = retrieve_memory(conn, mem[0])
                store_in_db(conn, session_id, mem[0], val, "neutral")  # Updated to use store_in_db
    except Exception as e:
        print(f"Error summarizing session: {e}")

def fetch_recent_memory_context(conn, user_input):
    try:
        similar = retrieve_similar_memories(user_input, conn)
        return "\n".join(similar) if similar else "No relevant memories found."
    except Exception as e:
        print(f"Error fetching memory context: {e}")
        return "Error retrieving memory context."

def add_empathetic_tone(response):
    # Implementation to add an empathetic tone to the response
    empathetic_phrases = [
        "I understand how you feel.",
        "That sounds really tough.",
        "I'm here for you.",
    ]
    # Choose a random empathetic phrase to prepend
    empathetic_phrase = random.choice(empathetic_phrases)
    return f"{empathetic_phrase} {response}"
