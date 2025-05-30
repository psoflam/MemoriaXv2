from memoriax2.nlp.processor import analyze_input  # Import the analyze_input function from the memoriax2.nlp.processor module
from memoriax2.storage.database import mark_confirmed, store_in_db, store_memory, retrieve_memory  # Import store_memory and retrieve_memory functions from the memoriax2.storage.database module
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.nlp.memory_recall import retrieve_similar_memories, embed_text
import faiss
import numpy as np
import random
import uuid
from memoriax2.memory.index_engine import MemoryIndex, memory_index

memory_index = MemoryIndex(384)  # or whatever your embedding dimension is

# Add startup log
print("MemoryIndex initialized with:", len(memory_index), "items")

def process_input(user_input, conn, session_id, memory_index):
    try:
        context = fetch_recent_memory_context(conn, user_input, memory_index)
        emotion = detect_emotion(user_input)
        response = chat_with_user(user_input, context, emotion)

        # Generate a memory key
        generated_key = f"entry_{uuid.uuid4()}"

        # Log the turn
        store_in_db(conn, session_id, generated_key, response, emotion)

        # Store memory explicitly
        store_memory(conn, generated_key, user_input, response, memory_index)

        # After storing memory, print index status and list keys
        memory_index.print_index_status()
        print("Current keys:", memory_index.list_keys())

        return response, emotion
    except Exception as e:
        print(f"Error processing input: {e}")
        return "I'm sorry, something went wrong.", None

def store_memory(conn, key, user_input, response, memory_index):
    # Create the vector from the user input (not the bot response)
    new_embedding = embed_text(user_input)

    # Debug print to check the type and dtype of new_embedding
    print(f"[DEBUG] Type of new_embedding: {type(new_embedding)}, dtype: {getattr(new_embedding, 'dtype', None)}")

    cursor = conn.cursor()

    # Similarity deduplication logic (optional)
    cursor.execute("SELECT key, embedding FROM memory_embeddings ORDER BY rowid DESC LIMIT 20")
    recent = cursor.fetchall()

    for recent_key, emb_blob in recent:
        existing = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(new_embedding, existing) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing))
        if sim > 0.9:
            print(f"Skipping memory '{key}' due to similarity with '{recent_key}'")
            return

    # Store full memory content
    cursor.execute("INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (key, user_input))
    cursor.execute("INSERT OR REPLACE INTO memory_embeddings (key, embedding) VALUES (?, ?)", (key, new_embedding.tobytes()))
    memory_index.add_memory(key, new_embedding)

    print("Top FAISS keys now:", memory_index.list_keys()[:3])
    conn.commit()
    
   

def generate_response(user_input, conn=None):
    emotion = detect_emotion(user_input)
    context = ""  # placeholder for memory/context logic
    base = generate_base_response(user_input + " " + context)

    if emotion == "sad":
        response = make_response_more_empathetic(base)
    else:
        response = f"Calmly, {base}"

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

def fetch_recent_memory_context(conn, user_input, memory_index):
    try:
        similar = retrieve_similar_memories(user_input, conn, memory_index)
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

def retrieve_similar_memories(input_text, conn, memory_index, top_k=3, recent_memory_limit=5):
    try:
        # Embed the input text
        input_embedding = embed_text(input_text)
        
        # Add a type check for the embedding
        if not isinstance(input_embedding, np.ndarray):
            print("[EMBEDDING ERROR] Failed to embed input_text:", input_text)
            return []

        # Query MemoryIndex instead of calculating cosine similarities manually
        top_memory_ids = memory_index.query_similar(input_embedding, top_k)

        if not top_memory_ids:
            print("No results from FAISS")
            return []

        # Fetch memory texts and keys from the database using the retrieved IDs
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.key, mem.value 
            FROM memory_embeddings m 
            JOIN memory mem ON m.key = mem.key 
            WHERE m.key IN ({})
        """.format(",".join("?" for _ in top_memory_ids)), top_memory_ids)
        top_memories = cursor.fetchall()

        # Prioritize memories with matching emotional tone
        input_emotion = detect_emotion(input_text)
        prioritized_memories = [
            mem for mem in top_memories 
            if detect_emotion(mem[1]) == input_emotion and mem[0].strip().lower() != "exit"
        ]

        # Limit repetition: Exclude recently used memories
        cursor.execute("SELECT key FROM recent_memories ORDER BY timestamp DESC LIMIT ?", (recent_memory_limit,))
        rows = cursor.fetchall()
        recent_memories = {row[0] for row in rows} if rows else set()
        final_memories = [mem for mem in prioritized_memories if mem[0] not in recent_memories]

        return final_memories
    except Exception as e:
        print(f"Error retrieving similar memories: {e}")
        return []

def load_index_from_db(self, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT key, embedding FROM memory_embeddings")
    rows = cursor.fetchall()

    for idx, (key, blob) in enumerate(rows):
        vec = np.frombuffer(blob, dtype=np.float32)
        self.add_memory(key, vec)

    print(f"[MemoryIndex] Loaded {len(rows)} items from DB into FAISS index.")
