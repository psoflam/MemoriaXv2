from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
import json

# Load the tokenizer and model
model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load the training data
with open('data/train.json', 'r') as f:
    training_data = json.load(f)

# Tokenize the input data
inputs = tokenizer([
    f"[PERSONA] [CONTEXT] User: {item['input']} Emotion: {item['emotion']} → MemoriaX: {item['response']}"
    for item in training_data
], return_tensors='pt', padding=True, truncation=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs['input_ids'],
    eval_dataset=inputs['input_ids'],
    tokenizer=tokenizer
)

# Train the model
trainer.train() 