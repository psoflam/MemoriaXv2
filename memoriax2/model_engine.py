from llama_cpp import Llama

# Initialize the LLaMA model
llm = Llama(model_path="memoriax2/models/mistral-7b-instruct-v0.1.Q4_0.gguf", n_ctx=2048)

def generate_response(prompt: str) -> str:
    """
    Generates a response based on the given prompt using the LLaMA model.
    """
    response = llm(prompt, max_tokens=150, temperature=0.7)
    return response["choices"][0]["text"].strip() 