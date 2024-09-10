from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum", from_tf=True)

# Load and prepare dataset
def preprocess_function(examples):
    inputs = tokenizer(examples['prompt'], max_length=512, truncation=True, padding='max_length')
    outputs = tokenizer(examples['text'], max_length=512, truncation=True, padding='max_length')
    inputs['labels'] = outputs['input_ids']
    return inputs

# Load your dataset from local csv file 
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'validation.csv'})
dataset = dataset.map(preprocess_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')
