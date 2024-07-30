# Introduction
The data contains questions & answers from Quora. The prerequisite is to understand the nature of data before development of the AI model can commence. Relevant literature survey was conducted, which is mentioned in the final report. Following is the summary of the case study.

### Understanding Data & Analysis: 
* The data has two features “question” & “answer”. 
* 56402 rows exist.
* Data is written mostly in English.
* Data is not cleaned as it is written by various authors.
* Data has significant number of emoticons.
* Shorthand is used extensively.
### Data Cleaning:
* Removed symbols & emoticons using regular expression library.
* Used Lemmatization.
* Tokenization for preprocessing.
### Environment Setup:
* Running the model on local system was taking more than 77 hours for GPT.
* Used Amazon EC2 instance (p3.2xlarge) with Nvidia Tesla-V100 16GB. 
* p3.2xlarge reduced the execution time by 99.02 %.
* Used CUDA for running Pytorch on GPU.
### Model Selection & Training:
* Used Llama, GPT, BERT & T5 for fine tuning the model.
* Split the data in 80/20 ratio for training & testing.
### Evaluation & Testing:
* Using the standard metric like ROUGE, BLEU, and F1-score.
### Insights & Visualization:
* Plotted graph & heat maps for comparison of model performance.
* Interesting insights for the given problem in detailed report.

Full detailed report & presentation available at:([GitHub])
(https://github.com/ChetanSharan/nlp-models-collection/blob/main/documentation/)
