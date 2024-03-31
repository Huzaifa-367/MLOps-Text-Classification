Text Classification with Naive Bayes
This repository contains code for a text classification model trained using the Naive Bayes algorithm. The model is trained on a dataset of text samples labeled with different categories, and it can predict the category of new text inputs.

Dataset
The model is trained on a custom dataset containing text samples labeled with different emotions: happy, sad, angry, and neutral. Each text sample is associated with a label indicating the emotion it represents.

Model Training
The training process involves the following steps:

Data Preprocessing: The text data is preprocessed to remove punctuation, URLs, and convert the text to lowercase. Additionally, lemmatization is applied to normalize the words.
Tokenization: The preprocessed text data is tokenized using the BERT tokenizer, which converts the text into numerical tokens suitable for input to the model.
Model Training: The tokenized data is used to train a Multinomial Naive Bayes classifier. The classifier learns to predict the emotion labels based on the tokenized text features.
Model Testing
After training, the model is tested using a separate set of text samples. The testing process involves the following steps:

Preprocessing: Similar to training, the text samples are preprocessed to remove punctuation, URLs, and convert to lowercase.
Tokenization: The preprocessed text samples are tokenized using the same BERT tokenizer used during training.
Prediction: The trained model is used to predict the emotion labels for the tokenized text samples.
Evaluation: The predicted labels are compared with the true labels to evaluate the performance of the model in terms of accuracy and other metrics.
Usage
To use the model for predicting emotions in text, follow these steps:

Load the Model: Load the trained model from the provided file (naive_bayes_emotion_model.pkl).
Preprocess Text: Preprocess the input text by removing punctuation, URLs, converting to lowercase, and lemmatization.
Tokenization: Tokenize the preprocessed text using the BERT tokenizer.
Prediction: Use the trained model to predict the emotion label for the tokenized text.
Output: The predicted emotion label can be used for further analysis or applications.
Requirements
Python 3.x
Libraries: scikit-learn, nltk, transformers
Credits
The model training and testing code is developed by [Your Name].
The dataset used for training is collected from [[Source](https://huggingface.co/datasets/dair-ai/emotion)].
