from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset, DatasetDict
import json
import torch
import os

# Load the tokenizer and model
model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token to eos_token (common workaround for GPT-style models)
tokenizer.pad_token = tokenizer.eos_token

# Alternatively, if you want to use a custom [PAD] token:
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained(model_name)

TRAIN_PATH = "train.json"
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"{TRAIN_PATH} not found. Make sure your dataset generation script succeeded.")

with open(TRAIN_PATH, "r") as f:
    try:
        training_data = json.load(f)
        assert isinstance(training_data, list) and all("input" in x and "response" in x for x in training_data)
    except Exception as e:
        raise ValueError(f"Invalid training data format: {e}")

# Format into strings
formatted_data = [
    {
        "text": f"[PERSONA] [CONTEXT] User: {item['input']} Emotion: {item['emotion']} → MemoriaX: {item['response']}"
    }
    for item in training_data
]

# Create HF-compatible dataset
hf_dataset = Dataset.from_list(formatted_data)

# Tokenize with map
tokenized_dataset = hf_dataset.map(
    lambda example: tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    ),
    batched=True
)

# Add labels (same as input_ids for GPT-style causal LM)
tokenized_dataset = tokenized_dataset.map(
    lambda x: {"labels": x["input_ids"]},
    batched=True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train() 