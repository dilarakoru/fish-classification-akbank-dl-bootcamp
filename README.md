# fish-classification-akbank-dl-bootcamp

## Overview
This repository contains a fish classification project developed as part of the Akbank Deep Learning Bootcamp. The goal of this project is to classify images of different fish species using an Artificial Neural Network (ANN) model. The entire training and evaluation processes were conducted on Kaggle, utilizing its cloud environment for computational resources and dataset management.

## Project Highlights
- **Project Type:** Image Classification
- **Model Type:** Artificial Neural Network (ANN)
- **Framework:** TensorFlow, Keras
- **Data Source:** Kaggle datasets (no local data storage required)
- **Training Platform:** Kaggle (links to training notebooks provided)
- **Fish Species:** 9 classes including House Mackerel, Red Mullet, Trout, and more.

## Dataset
The dataset used in this project is hosted on Kaggle, consisting of various images of fish species. These images were resized to 224x224 pixels to ensure consistency for training the model. The dataset was split into training, validation, and test sets:
- **Training Data:** 80% of the dataset
- **Validation Data:** 5% of the dataset
- **Test Data:** 15% of the dataset

## Model Architecture

The artificial neural network (ANN) model was constructed using TensorFlow and Keras, following a similar core structure in both versions of the project. The key details of the architecture include:

- Input Layer: Accepts 224x224 RGB images, flattening them to make them compatible with fully connected layers.
- Hidden Layers: A series of Dense layers with ReLU activation functions, coupled with Batch Normalization and Dropout layers for regularization to prevent overfitting.
- Output Layer: A Softmax layer with 9 units, each representing a different fish species.
- Optimizer: Adam optimizer, with learning rates that vary between the two versions.
- Loss Function: Categorical Crossentropy, suitable for multi-class classification.
- Metrics: Accuracy, used to track the model's performance during training.
### Key Differences Between the Two Project Versions
While the underlying model architecture remains consistent across both project versions, the hyperparameters used for training differ, leading to variations in the training dynamics and results. Here’s a breakdown of the differences:

#### Version 1: Training with Larger Batch Size and Learning Rate

- Batch Size: 64
- Learning Rate: 0.001
- Model Characteristics: A higher batch size allows the model to process more images at once, which can speed up training but might result in less frequent updates. Combined with a higher learning rate, the model adjusts weights more aggressively during training, which can help reach an optimal solution faster but might risk overshooting the optimal point.
-  Training Behavior: This setup may be more prone to overfitting, but can achieve higher accuracy quickly if the data is well-represented in each batch.
### Version 2: Training with Smaller Batch Size and Learning Rate

- Batch Size: 32
- Learning Rate: 0.0001
- Model Characteristics: A smaller batch size means that the model updates weights more frequently, but with fewer samples per update, making the training process potentially longer but more precise. A smaller learning rate means that each update is smaller, allowing the model to make finer adjustments as it learns.
- Training Behavior: This setup is more likely to avoid overfitting due to the smaller updates, potentially leading to a more generalized model that performs better on unseen data, though training time might be longer.
#### Model Performance
The model achieved the following results:
- **Test Accuracy:** 88.3%
- **Test Loss:** 0.6431
- **Evaluation Metrics:** Confusion Matrix, Precision, Recall, and F1-Score.

## Notebooks
- [Notebook with Larger Batch Size and Learning Rate](https://www.kaggle.com/code/dilarakoru/akbank-dl-bootcamp?scriptVersionId=203132715)
- [Notebook with Smaller Batch Size and Learning Rate](https://www.kaggle.com/code/dilarakoru/akbank-dl-bootcamp2)

