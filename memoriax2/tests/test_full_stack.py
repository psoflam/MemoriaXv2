from memoriax2.core.chatbot import process_user_input

# Test cases to verify the full stack functionality

def test_store_fact():
    user_input = "My name is Colin."
    response = process_user_input(user_input)
    assert "Stored name: Colin" in response, "Fact storage failed."


def test_retrieve_fact():
    user_input = "Do you know my name?"
    response = process_user_input(user_input)
    assert "Your name is Colin" in response, "Fact retrieval failed."


def test_recall_memory():
    user_input = "What's your favorite memory of me?"
    response = process_user_input(user_input)
    assert "favorite memory" in response, "Memory recall failed."


def test_construct_prompt():
    user_input = "Tell me about my past."
    response = process_user_input(user_input)
    assert "Context:" in response, "Prompt construction failed."


def test_generate_response():
    user_input = "How are you?"
    response = process_user_input(user_input)
    assert response, "Response generation failed."


if __name__ == "__main__":
    test_store_fact()
    test_retrieve_fact()
    test_recall_memory()
    test_construct_prompt()
    test_generate_response()
    print("All tests passed!") 