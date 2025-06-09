def build_prompt(user_input, recalled_memories):
    """
    Constructs a prompt for the model based on user input and recalled memories.
    """
    # Example prompt construction logic
    prompt = "User said: " + user_input + "\n"
    if recalled_memories:
        prompt += "Recalled memories: " + ", ".join(recalled_memories) + "\n"
    prompt += "Respond accordingly."
    return prompt 