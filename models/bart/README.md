# Installation & Execution Instruction for BART
This ReadMe is in conjection with the primary [README](../../README.md) file. Please refer that file for basic information & structure of the project.

## Setting up Envirnment
* Make sure you have Python 3.12 & PIP version 24 installed.
* Setup a virtual environment in python & activate it using following commands.

`cd models/bart`

`pip install virtualenv`

`python -m venv ~/env/bart`

`source ~/env/bart/bin/activate`

* PIP all the required libraries before running the code, using the following command.

`pip install nltk datasets evaluate numpy transformers`

* To train the model, execute the following command. It takes around two hours to train on the p3.2xlarge with GPU Tesla V100 16GB.

`python train.py`

* To run the model, execute the following command. 

`python run.py`

* The program will promt for questions & other instruction.
