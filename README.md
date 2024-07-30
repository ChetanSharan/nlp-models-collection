# An AI Model for NLP | A Case Study
This is a case study developed for comparing multiple AI Models for a same data set. The data is hosted on [HuggingFace](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset) website.

## Methodology
The data contains questions & answers from Quora. The prerequisite is to understand the nature of data before development of the AI model can commence. Relevant literature survey was conducted, which is mentioned in the final report. Following is the summary of the case study.

### Step 1: Understanding Data & Analysis: 
* The data has two features “question” & “answer”. 
* 56402 rows exist.
* Data is written mostly in English.
* Data is not cleaned as it is written by various authors.
* Data has significant number of emoticons.
* Shorthand is used extensively.
### Step 2: Data Cleaning:
* Removed symbols & emoticons using regular expression library.
* Used Lemmatization.
* Tokenization for preprocessing.
### Step 3: Environment Setup:
* Running the model on local system was taking more than 77 hours for GPT.
* Used Amazon EC2 instance (p3.2xlarge) with Nvidia Tesla-V100 16GB. 
* p3.2xlarge reduced the execution time by 99.02 %.
* Used CUDA for running Pytorch on GPU.
### Step 4: Model Selection & Training:
* Used Llama, GPT, BERT & T5 for fine tuning the model.
* Split the data in 80/20 ratio for training & testing.
### Step 5: Evaluation & Testing:
* Using the standard metric like ROUGE, BLEU, and F1-score.
### Step 6: Insights & Visualization:
* Plotted graph & heat maps for comparison of model performance.
* Interesting insights for the given problem in detailed report.

## Tech Stack
Following is the detailed description of the technology used for the project.
### Language & Libraries
* Python 3.11.4.
* NLTK, Datasets, NumPy, Evaluate, Transformers etc.
### Cloud Stack (Tested On)
* Providor: AWS
* Compute: EC2
* Instance type : p3.2xlarge 
* GPU : Tesla V100 16 GB VRAM
### Front End
* Command Line Interface
### NLP Models Used
* T5 (Text-to-Text Transfer Transformer)
* GPT (Generative pre-trained transformers)
* Llama (Large Language Model Meta AI)
Full detailed report & presentation available at: 
[Here](https://github.com/ChetanSharan/nlp-models-collection/blob/main/documentation)
