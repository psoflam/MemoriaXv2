from memoriax2.core.chatbot import process_user_input
from memoriax2.shared.utils_db import create_connection
from memoriax2.memory.index_engine import MemoryIndex

# Initialize database connection and memory index
conn = create_connection("memoriax2/data/fact_memory.db")
memory_index = MemoryIndex()
memory_index.load_index_from_db(conn)

# Initialize session_id
session_id = "test_session"

# Test cases to verify the full stack functionality

def test_store_fact():
    user_input = "My name is Colin."
    process_user_input(user_input, conn, session_id, memory_index)

    # Query memory table to confirm storage
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM memory WHERE value LIKE '%Colin%'")
    result = cursor.fetchone()

    assert result is not None, "Fact storage failed — memory not saved."
    print("[TEST PASSED] Memory fact about name was saved.")


def test_retrieve_fact():
    user_input = "Do you know my name?"
    response_text, _ = process_user_input(user_input, conn, session_id, memory_index)
    assert "colin" in response_text.lower(), "Name not mentioned in response."
    print("[TEST PASSED] Fact retrieval successful.")


def test_recall_memory():
    user_input = "What's your favorite memory of me?"
    response = process_user_input(user_input, conn, session_id, memory_index)
    assert "favorite memory" in response, "Memory recall failed."


def test_construct_prompt():
    user_input = "Tell me about my past."
    response = process_user_input(user_input, conn, session_id, memory_index)
    assert "Context:" in response, "Prompt construction failed."


def test_generate_response():
    user_input = "How are you?"
    response = process_user_input(user_input, conn, session_id, memory_index)
    assert response, "Response generation failed."


# Verify memory retrieval

def test_fact_inserted():
    user_input = "My name is Colin."
    process_user_input(user_input, conn, session_id, memory_index)

    results = retrieve_similar_memories("What is my name?", conn, memory_index)
    assert any("Colin" in val for _, val in results), "Memory retrieval failed — no memory containing 'Colin'"
    print("[TEST PASSED] Memory containing 'Colin' retrieved successfully.")


def test_prompt_construction():
    user_input = "Do you know my name?"
    response_text, _ = process_user_input(user_input, conn, session_id, memory_index)
    assert "Context:" in response_text, "Prompt construction failed."
    print("[TEST PASSED] Prompt construction successful.")


if __name__ == "__main__":
    test_store_fact()
    test_retrieve_fact()
    test_recall_memory()
    test_construct_prompt()
    test_generate_response()
    test_fact_inserted()
    test_prompt_construction()
    print("All tests passed!") 