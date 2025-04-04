
# Mental Health Classification using BERT 🤖🧠

Welcome to the **Mental Health Classification** project – an end-to-end solution that leverages Hugging Face's BERT to detect mental health states from text. This repository contains code for data preprocessing, model fine-tuning, evaluation, and a user-friendly Streamlit app for real-time predictions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Preprocessing & Analysis](#data-preprocessing--analysis)
- [Model Fine-Tuning](#model-fine-tuning)
- [Model Evaluation](#model-evaluation)
- [Streamlit App](#streamlit-app)
- [Dataset](#Dataset)


---

## Overview

In today's world, mental health is more important than ever. This project aims to classify text statements into various mental health categories (e.g., Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal) using a fine-tuned BERT model. The model is trained on a curated dataset of mental health-related sentences and is evaluated to achieve an overall accuracy of **90%**. Although performance on some categories (like Depression and Suicidal) could be improved, the results are promising.

---

## Features

- **Data Cleaning & Preprocessing:**  
  - Removal of special characters, numbers, and stopwords using NLTK.
  - Visualization of data distribution with pie charts.

- **Data Balancing:**  
  - Uses `RandomOverSampler` from `imblearn` to handle class imbalance.

- **Model Training:**  
  - Fine-tuning of BERT for sequence classification with Hugging Face's `Trainer` and `TrainingArguments`.
  - Configurable training parameters (learning rate, batch sizes, epochs, etc.).

- **Model Evaluation:**  
  - Generates a detailed classification report and confusion matrix.
  - Achieves 90% overall accuracy, with high precision and recall in most categories.

- **Real-Time Prediction:**  
  - A detection system that processes a user’s input text to predict the mental health state.
  
- **User Interface:**  
  - A sleek Streamlit web app with custom styling and emojis for an engaging user experience.

---

## Data Preprocessing & Analysis

- **Data Loading & Cleaning:**  
  The raw dataset is loaded from a CSV file, and unnecessary columns and null values are removed. Text statements are cleaned using a custom function that converts text to lowercase, strips special characters, and removes stopwords.

- **Exploratory Data Analysis:**  
  Visualizations such as pie charts provide insights into the distribution of mental health conditions within the dataset.

- **Addressing Imbalance:**  
  The `RandomOverSampler` is used to ensure a balanced distribution across different classes.

---

## Model Fine-Tuning

The BERT model is fine-tuned for mental health classification using Hugging Face's `Trainer` and `TrainingArguments`. Key training parameters include:

- **Epochs:** 10
- **Learning Rate:** 2e-5
- **Batch Size:** 16 (per device)
- **Warmup Steps:** 250
- **Logging & Checkpointing:** Configured to save checkpoints and log metrics after each epoch

The code is modularized into cells, making it easy to run in a Jupyter Notebook or as a standalone script.

---

## Model Evaluation

After training, the model is evaluated using a test split. The evaluation metrics include:

- **Classification Report:**  
  Displays precision, recall, and F1-scores for each mental health category.
  
- **Confusion Matrix:**  
  Visualized with Seaborn to show the distribution of correct and misclassified examples.

Overall, the model performs well with an accuracy of 90%, although further improvements are needed for the Depression and Suicidal categories.

---

## Streamlit App

The project includes a user-friendly Streamlit web app that allows users to input a sentence and receive a mental health state prediction in real time. The app features:

- A custom-designed interface with cool emojis and hover effects.
- Real-time predictions with confidence scores.
- An engaging and clean layout.


## 📂 Dataset
The dataset used for training is **Sentiment Analysis for Mental Health**, available on Kaggle:

📥 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)

