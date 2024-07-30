# Load the model for evaluation or further use
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained('./fine-tuned-model')
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model')

# Initialize a pipeline for text generation
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Test the model
response = generator("What is proxy?", max_length=50)
print(response)