Sentiment Analysis Using Neural Networks
This project implements a sentiment analysis model to classify restaurant reviews as positive or negative. It utilizes text preprocessing, word embeddings, and a neural network built with TensorFlow/Keras to achieve efficient sentiment classification.

Introduction
Sentiment analysis is a natural language processing task that determines the sentiment expressed in a text. This project focuses on restaurant reviews to classify sentiments into positive or negative, helping businesses analyze customer feedback effectively.

Features
-Preprocessing of raw text data (cleaning, tokenization, lemmatization).
-Uses word embeddings for representing text.
-Customizable neural network model with:
-Embedding layer
-Multiple hidden layers with Dropout for regularization
-Early stopping mechanism to prevent overfitting.
-Generates performance metrics like accuracy, classification report, and a confusion matrix.
-Allows users to input custom reviews for sentiment prediction.

🛠 Tech Stack
Programming Languages:
-Python 🐍
Frameworks & Libraries:
-TensorFlow & Keras
-NumPy
-NLTK for text preprocessing and lemmatization
-Matplotlib for visualization

Model Architecture
The model architecture consists of:

1.Embedding Layer:
-Converts words into dense vector representations.
2.Global Max Pooling:
-Reduces dimensionality and retains the most relevant features.
3.Dense Layers:
-Fully connected layers with ReLU activation for learning complex patterns.
4.Dropout Regularization:
-Prevents overfitting by randomly disabling neurons during training.
5.Output Layer:
-A single neuron with sigmoid activation to classify sentiment.

📧 Contact
Email: priyadarshinidebamita@gmail.com
LinkedIn: https://www.linkedin.com/in/debamita-priyadarshini-1b7841296/?

