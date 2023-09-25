import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class Regression(object):
    def __init__(self):
        """
        Initialize instance of OneHotEncoder in self.oh for use in onehot function.
        """
        self.oh = OneHotEncoder()

    def onehot(self, labels):
        """
        Helper function to encode a labels into one hot encoding format

        Args:
                labels: list of class labels

        Return:
                onehotencoded: (N, C) numpy array where:
                        N is the number of datapoints in the list 'labels'
                        C is the number of distinct labels/classes in 'labels'

        Hints: 
            1. np.array may be helpful in converting a sparse matrix into a numpy array
            2. Recall that fit_transform takes in a 2d array input instead of a 1d array input

        """
        
        labels = np.array(labels).reshape(-1,1)
        onehotencoded = self.oh.fit_transform(labels)
        return onehotencoded.toarray()

    def gradient(self, X, Y, W):
        """
        Apply softmax function to compute the predicted labels and calculate the gradients of the loss w.r.t the weights weights.

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                W: (D, C) numpy array of weights.

        Return:
                gradient: (D,C) numpy array of the computed gradients

        Hint: Use the formula in Section 1.1 of HW2.ipynb to compute the gradients
        """

        mu = 0.01  # Do not change momentum value.

        # STEP 1: calculate the product of data
        product = np.dot(X, W)

        # STEP 2: take the softmax of each row
        softmax = np.exp(product) / np.sum(np.exp(product), axis=1, keepdims=True)
        
        # STEP 4: calculate the gradients of the loss w.r.t the weights
        gradient = - 1 * np.dot(X.T, (Y - softmax)) / X.shape[0] + 2 * mu * W
        return gradient

    def gradient_descent(self, X, Y, epochs=10, eta=0.1):
        """
        Basic gradient descent algorithm with fixed eta and mu

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                epochs: Number of epochs for the gradient descent (optional, defaults to 10).
                eta: Learning rate (optional, defaults to 0.1)

        Return:
                weight: (D,C) weight matrix

        Hint: Weight should be initialized to be zeros
        """
        W = np.zeros((X.shape[1], Y.shape[1]))
        for _ in range(epochs):
                W = W - (eta * self.gradient(X, Y, W))
        return W


    def fit(self, data, labels):
        """
        Fit function for calculating the weights using gradient descent algorithm.
        NOTE : This function is given and does not have to be implemented or changed.

        Args:
                data: (N, D) TF-IDF features for the data.
                labels: (N, ) list of class labels
        """

        X = np.asarray(data)
        Y_onehot = self.onehot(labels)
        self.W = self.gradient_descent(X, Y_onehot)

    def predict(self, data):
        """
        Predict function for predicting the class labels.

        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                predictedLabels: (N,) list of predicted classes for the data.
        """
        # Represent the data as a numpy matrix
        X = np.asarray(data)

        # Use the predicted weights to compute the probability of each class
        predictedLabels = np.dot(X, self.W)

        # Take the argmax for each row and find the class with the highest probability
        predictedLabels = np.argmax(predictedLabels, axis=1)
        return predictedLabels

