# Set pad_token to eos_token (common workaround for GPT-style models)
tokenizer.pad_token = tokenizer.eos_token

# Alternatively, if you want to use a custom [PAD] token:
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})