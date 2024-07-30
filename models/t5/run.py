# Load the model for evaluation or further use
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model
# model = AutoModelForCausalLM.from_pretrained('./fine-tuned-model')
# tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model')


tokenizer = AutoTokenizer.from_pretrained("toughdata/flan-t5-base-quora-question-answer")
model = AutoModelForSeq2SeqLM.from_pretrained("toughdata/flan-t5-base-quora-question-answer")


# Initialize a pipeline for text generation
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Test the model
print("")
print("Hi, Welcome to Chetan's T5 Bot, you can ask any question and I will try to answer.")
print("Type 'quit' or 'q' to exit")
print("")
question = input("Type your question here: ")
while question.lower() != 'quit' and question.lower() != 'q':
    response = generator(question, max_length=100)
    print(response[0]['generated_text'])
    print("")
    question = input("You can ask more question here or type 'quit' or 'q' to exit:")
