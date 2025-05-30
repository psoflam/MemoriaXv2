from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-3750')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# Set up the text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the prompt
prompt = "[PERSONA] [CONTEXT] User: I'm feeling really low. Emotion: sad → MemoriaX:"

# Generate a response
response = generator(prompt, max_length=100, num_return_sequences=1)

# Print the generated text
print(response[0]['generated_text']) 