# ✍️ Handwritten Digit Recognition with Scikit-learn

This project demonstrates the use of a neural network (MLP) to recognize handwritten digits using the `digits` dataset from `sklearn.datasets`.

## 📄 Description

Using the `load_digits()` dataset from scikit-learn, we build and train a Multi-layer Perceptron (MLP) classifier to recognize 8x8 pixel grayscale images of handwritten digits (0–9). The dataset is preprocessed, visualized, and split into training and test sets.

## 📁 Dataset

- Dataset: `sklearn.datasets.load_digits()`
- Each image is 8x8 pixels (grayscale)
- Labels: Digits from 0 to 9
- Size: ~1,800 images

## 📊 Visualization

- Grid of 16 digits plotted using `matplotlib`
- Each digit image is displayed with its corresponding label

## 🔧 Preprocessing

- Reshaped image data from 2D (8x8) to 1D (64)
- Split into training (1000 samples) and test sets (remainder)

## 🧠 Model

- Model: `MLPClassifier` from `sklearn.neural_network`
- Hidden layers: 1 hidden layer with 15 neurons
- Activation: `logistic`
- Solver: `sgd` (stochastic gradient descent)
- Learning rate: 0.1
- Epoch loss visualized using a loss curve

## 🧪 Evaluation

- Predicted labels vs actual labels
- Metric used: `accuracy_score` from `sklearn.metrics`

## 🚀 How to Run

1. Open the notebook `Recognizing_Hand_Written_Digits_in_Scikit_Learn.ipynb`
2. Run all cells step by step to:
   - Load and visualize data
   - Train an MLP model
   - Plot training loss
   - Evaluate predictions

## ✅ Requirements

- Python 3.x
- scikit-learn
- numpy
- matplotlib

## 📌 Notes
This project is a simple example of image recognition using classical machine learning (no deep learning libraries like TensorFlow or PyTorch involved).
