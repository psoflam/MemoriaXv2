from memoriax2.nlp.processor import analyze_input  # Import the analyze_input function from the memoriax2.nlp.processor module
from memoriax2.storage.database import store_memory, retrieve_memory  # Import store_memory and retrieve_memory functions from the memoriax2.storage.database module
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.nlp.memory_recall import retrieve_similar_memories
import faiss

def process_input(user_input, conn):
    entities = analyze_input(user_input)  # Analyze the user input to extract entities
    for entity, label in entities:
        if label == 'GPE':  # Check if the entity is a geopolitical entity (like a location)
            store_memory(conn, 'location', entity)  # Store the location entity in the database
            return f"Got it! You're in {entity}."  # Return a response acknowledging the location
    
    last_location = retrieve_memory(conn, 'location')  # Retrieve the last stored location from the database
    if last_location:
        return f"You're in {last_location}, right?"  # Return a response with the last known location
    
    # Retrieve similar memories
    similar_memories = retrieve_similar_memories(user_input, conn)
    # Inject similar memories into the response context
    context = " ".join(similar_memories)
    
    # Generate response with context
    response = generate_response(user_input, context)
    return response

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
