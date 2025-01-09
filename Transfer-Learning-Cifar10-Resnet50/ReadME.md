# CIFAR-10 Image Classification using ResNet-50

This repository contains a deep learning project that implements a ResNet-50-based model for classifying images in the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 images distributed across 10 classes. This project achieves **94.66% accuracy** on the test data using transfer learning and custom training.

---

## Overview

The project includes:
- **Dataset Preparation**: Reduced dataset for training and separate unused data for testing.
- **Model Architecture**: ResNet-50 pre-trained on ImageNet with additional layers for CIFAR-10 classification.
- **Training Strategy**: Early stopping and model checkpointing for optimized training.
- **Submission**: Predictions saved for Kaggle submission in the required format.

---

## Features

- **Dataset Preprocessing**:
  - Reduced CIFAR-10 dataset to 40,000 images for training, stratified across classes.
  - A separate folder, `for_test_dir`, contains unused images for model testing.

- **Model Details**:
  - ResNet-50 backbone pre-trained on ImageNet.
  - Upsampling and fully connected layers for CIFAR-10 image classification.
  - Regularization techniques like dropout to prevent overfitting.

- **Evaluation Results**:
  - **Loss**: 0.2436
  - **Accuracy**: 94.66%

- **Kaggle Submission**:
  - Creates a `submission.csv` file with image IDs and corresponding predictions.

---

## Setup

### Prerequisites
- Python >= 3.8
- TensorFlow >= 2.5
- Pandas, NumPy, Matplotlib, and scikit-learn


