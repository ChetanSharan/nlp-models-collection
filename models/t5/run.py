# Load the model for evaluation or further use
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained('./fine-tuned-model')
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model')

# Initialize a pipeline for text generation
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Test the model
print("Hi, Welcome to Chetan's T5 Bot, you can ask any question and I will try to answer.")
print("Type 'quit' or 'q' to exit")
print("")
question = input("Type your question here: ")
while question.lower() != 'quit' and question.lower() != 'q':
    response = generator(question, max_length=50)
    print(response)
    print("")
    question = input("You can ask more question here or type 'quit' or 'q' to exit:")
