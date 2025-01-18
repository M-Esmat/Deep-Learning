# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
DATA_PATH = r'Sentiment140_dataset\train.csv'
MAXLEN = 100
VOCAB_SIZE = 50000
EMBEDDING_DIM = 100

# Load and preprocess data
def load_data(path):
    """Load the dataset from the given path."""
    data = pd.read_csv(path, encoding='ISO-8859-1', header=None)
    data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    data['target'] = data['target'].map({0: 0, 4: 1})  # Map target labels
    return data

def preprocess_text(text, lemmatizer, stopwords):
    """Preprocess text by cleaning, tokenizing, and lemmatizing."""
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Build and train models
def build_lstm_model(vocab_size, embedding_dim, input_length):
    """Build an LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(128),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(vocab_size, embedding_dim, input_length):
    """Build a Bidirectional LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        Bidirectional(LSTM(128)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main function
def main():
    # Load and preprocess data
    data = load_data(DATA_PATH)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    data['clean'] = data['text'].apply(lambda x: preprocess_text(x, lemmatizer, stopwords_set))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['clean'], data['target'], test_size=0.2, random_state=42, stratify=data['target'])

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=MAXLEN)
    X_test_padded = pad_sequences(X_test_seq, maxlen=MAXLEN)

    # Build and train LSTM model
    lstm_model = build_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)
    lstm_history = lstm_model.fit(X_train_padded, y_train, epochs=2, batch_size=32, validation_data=(X_test_padded, y_test))
    plot_training_history(lstm_history)

    # Build and train Bidirectional LSTM model
    bilstm_model = build_bilstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)
    bilstm_history = bilstm_model.fit(X_train_padded, y_train, epochs=2, batch_size=64, validation_data=(X_test_padded, y_test))
    plot_training_history(bilstm_history)

if __name__ == "__main__":
    main()
