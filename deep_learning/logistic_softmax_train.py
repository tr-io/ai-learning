# training log regression w/ softmax
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

def y2indicator(y, K):
    # get indicator matrix from targets
    # K = # of classes
    # y = targets
    # one-hot encoding stuff
    # N samples, K classes
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# get preprocessed data
X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32) # labels

D = X.shape[1] # number of features
K = len(set(Y)) # number of classes - cast Y as a set so no repeated classes are counted

# split data set into train and test sets
Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K) # this is the target matrix, T (from eqns)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

# initialize random weights
W = np.random.randn(D, K)
b = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY))

# now we can train
train_costs = []
test_costs = []
lr = 0.001

for i in range(10000):
    # use activation functions to forward values to hidden layer
    # e.g. 'go through neural network'
    # pY vals are predicted values
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    # keep track of costs while doing gradient descent
    ctrain = cross_entropy(Ytrain_ind, pYtrain) # find loss (cost) of predicted val
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # now do gradient descent
    W -= lr * Xtrain.T.dot(pYtrain - Ytrain_ind)
    b -= lr * (pYtrain - Ytrain_ind).sum(axis=0)

    # for visualization
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification_rate: ", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate: ", classification_rate(Ytest, predict(pYtest)))

# now plot
legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()