from datasets import load_dataset
import evaluate
import nltk
import numpy as np

dataset = load_dataset("toughdata/quora-question-answer-dataset",split='train')
dataset = dataset.train_test_split(test_size=0.2)

# Load a tokenizer (adjust the model name to the one you are using)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Define a function to preprocess the data
def preprocess(data):
    inputs = data['question']
    targets = data['answer']
    return {'input_ids': tokenizer(inputs, padding='max_length', truncation=True, max_length=512)['input_ids'],
            'labels': tokenizer(targets, padding='max_length', truncation=True, max_length=512)['input_ids']}


# Preprocess the dataset
encoded_data = dataset.map(preprocess, batched=True)
print(encoded_data)


# Set up Rouge score for evaluation
nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    output_dir='./results',
    eval_strategy='epoch',
    num_train_epochs=2,
    save_strategy='epoch',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data['train'],
    eval_dataset=encoded_data["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()


# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-tokenizer')
