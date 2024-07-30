from datasets import load_dataset

train_data = load_dataset("toughdata/quora-question-answer-dataset", split="train")

# Load a tokenizer (adjust the model name to the one you are using)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define a function to preprocess the data
def preprocess(data):
    inputs = data['question']
    targets = data['answer']
    return {'input_ids': tokenizer(inputs, padding='max_length', truncation=True, max_length=512)['input_ids'],
            'labels': tokenizer(targets, padding='max_length', truncation=True, max_length=512)['input_ids']}


# Preprocess the dataset
encoded_data = train_data.map(preprocess, batched=True)


from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    output_dir='./results',
    eval_strategy='epoch',
    logging_dir='./logs',
    num_train_epochs=1,
    save_strategy='epoch',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data,
)

# Train the model
trainer.train()


# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')