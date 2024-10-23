import os
from flask import Flask, render_template, request
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch

app = Flask(__name__)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Llama 2 model and tokenizer (as a proxy for Llama 3)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Load a sample dataset
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:1000]")


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Set up LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare the model for fine-tuning
model = get_peft_model(model, lora_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=3,
    logging_steps=100,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving fine-tuned model...")
trainer.save_model("./llama2-finetuned-final")


# Function to summarize text using the fine-tuned model
def summarize_text(text, max_length=150, min_length=50, format_type="paragraph"):
    inputs = tokenizer(f"Summarize the following text in {format_type} format: {text}", return_tensors="pt",
                       max_length=512, truncation=True)
    summary_ids = model.generate(**inputs, max_length=max_length, min_length=min_length, length_penalty=2.0,
                                 num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        summary_type = request.form['summary_type']
        format_type = request.form['format_type']

        max_length = 150 if summary_type == 'short' else 300
        min_length = 50 if summary_type == 'short' else 100

        summary = summarize_text(text, max_length, min_length, format_type)
        return render_template('index.html', summary=summary)
    return render_template('index.html')


if __name__ == '__main__':
    # Fine-tune the model before starting the Flask app
    print("Fine-tuning the model...")
    trainer.train()
    trainer.save_model("./llama2-finetuned-final")
    print("Fine-tuning complete. Starting Flask app...")

    # Start the Flask app
    app.run(debug=True)