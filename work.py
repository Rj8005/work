import subprocess

# Installation on Google Colab
try:
    import google.colab
    subprocess.run(['python', '-m', 'pip', 'install', 'skorch', 'torch'])
except ImportError:
    pass

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Loading Data
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
mnist.data.shape

# Preprocessing Data
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.shape, y_train.shape

# Define a function to plot example images
def plot_example(X, y):
    plt.figure(figsize=(15, 15))

    for i in range(10):
        for j in range(10):
            index = i * 10 + j
            plt.subplot(10, 10, index + 1)
            plt.imshow(X[index].reshape(28, 28))
            plt.xticks([])
            plt.yticks([])
            plt.title(y[index], fontsize=8)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.show()

# Plot a selection of training images and their labels
plot_example(X_train, y_train)

import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim / 8)
output_dim = len(np.unique(mnist.target))

class ClassifierModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

net = NeuralNetClassifier(
    ClassifierModule,
    module__input_dim=mnist_dim,
    module__hidden_dim=hidden_dim,
    module__output_dim=output_dim,
    max_epochs=20,
    lr=0.1,
    device=device,
)

net.fit(X_train, y_train)

y_pred = net.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

error_mask = y_pred != y_test

# Plot examples of misclassified images
def plot_misclassified(X, y_true, y_pred):
    plt.figure(figsize=(15, 15))
    count = 0
    for i in range(len(y_true)):
        if count >= 100:
            break
        if y_true[i] != y_pred[i]:
            plt.subplot(10, 10, count + 1)
            plt.imshow(X[i].reshape(28, 28))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"True: {y_true[i]}\nPred: {y_pred[i]}", fontsize=8)
            count += 1

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.show()

plot_misclassified(X_test[error_mask], y_test[error_mask], y_pred[error_mask])
