import numpy as np
import cupy as cp
import pandas as pd
from sklearn.utils import shuffle

def init_weights_and_bias(M1, M2):
    """
    Takes in M1 (input size) and M2 (output size)
    Initializes weights of size M1 x M2
    Initializes bias of zeroes of size M2
    """
    W = cp.random.randn(M1, M2) / cp.sqrt(M1 + M2)
    b = cp.zeros(M2)
    return W.astype(cp.float32), b.astype(cp.float32)

# TODO: def init_filter - for CNNs
# TODO: def relu - for activation for neural network
# TODO: def softmax - softmax function

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def sigmoid_cost(T, Y):
    """
    Takes in Target (T) matrix and prediction (output) (Y) matrix
    Returns the cross_entropy cost
    Remember: only works with binary classification!
    """
    return -(T*cp.log(Y) + (1-T) * cp.log(1-Y)).sum()

# TODO: def cost/cost2 - general cross entropy function for softmax

def error_rate(targets, predictions):
    """
    Calculates the error rate of the predictions
    """
    return cp.mean(targets != predictions)



def getData(balance_ones=True, Ntest=1000):
    """
    Get's the facial expression data
    optional params:
    balance_ones: whether or not to handle the class imbalance
    Ntest = number of elements for test set
    """
    Y = []
    X = []
    i = 0
    for line in open('fer2013.csv'):
        if i == 0:
            i = 1
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X = cp.array(X) / 255.0
    Y = cp.array(Y)

    # shuffle data and split into training and test
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:]

    if balance_ones:
        # balance the 1 class since it has an imbalanced # of samples
        # used for logistic regression mostly but also when classifying all 7 labels
        X0, Y0 = Xtrain[Ytrain != 1, :], Ytrain[Ytrain != 1]
        X1 = Xtrain[Ytrain==1, :]
        X1 = cp.repeat(X1, 9, axis=0)
        Xtrain = cp.vstack([X0, X1])
        Ytrain = cp.concatenate((Y0, [1]*len(X1)))

    return Xtrain, Ytrain, Xvalid, Yvalid

# TODO: def getImageData - for convolution neural networks

def getBinaryData():
    X = []
    Y = []
    i = 0
    for line in open('fer2013.csv'):
        if i == 0:
            i = 1
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    
    return cp.array(X) / 255.0, cp.array(Y)