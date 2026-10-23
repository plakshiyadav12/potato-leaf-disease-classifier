# 🥔 Potato Leaf Disease Classifier

A deep learning project that aims to classify potato leaf images into three categories using a Convolutional Neural Network (CNN).

## 🎯 Project Objective

The goal is to build an image classification system that identifies:

- Early Blight

- Healthy

- Late Blight

The project uses TensorFlow and Keras for model development and Jupyter Notebook for experimentation.

## 🛠️ Tech Stack

- Python

- TensorFlow / Keras

- NumPy

- Matplotlib

- Jupyter Notebook

- Git and GitHub

## 📂 Project Structure

```
potato_leaf_disease_classifier/
├── training.ipynb
├── README.md
└── .gitignore
```

## 🧠 Development Progress

- Initial CNN architecture

- Initial model training and evaluation

- Identify and correct dataset class-label loading

- Retrain the model using the corrected dataset pipeline

- Evaluate predictions on unseen images

- Build an interactive web interface

- Document final evaluation results

## 📊 Dataset

The dataset contains separate Training, Validation, and Testing directories, each organized into the three target disease classes.

The dataset is excluded from this repository to avoid committing thousands of image files.

## 🚀 Planned Features

- Upload a potato leaf image

- Predict the corresponding disease category

- Display model confidence scores

- Visualize class probabilities

- Provide a clean, dark-themed interface

## ⚠️ Current Status

This project is under development. The initial model was trained with incorrectly configured dataset labels, and the classification pipeline is being corrected. The model's disease-classification performance has not yet been validated.

## 👩‍💻 Author

Developed as a machine learning and computer vision portfolio project.