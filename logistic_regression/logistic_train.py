# train the model for the ecommerce project
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

X, Y = get_binary_data()

# shuffle it so we don't overfit based on order
X, Y = shuffle(X, Y)

# split dataset into train and test
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1] # get the number of features (columns)
W = np.random.randn(D) # initialize random weights
b = 0 # bias term

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, Y):
    return -np.mean(T * np.log(Y) + (1-T) * np.log(1-Y))

train_costs = []
test_costs = []
lr = 0.001

for i in range(10000):
    y_train = forward(Xtrain, W, b)
    y_test = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, y_train)
    ctest = cross_entropy(Ytest, y_test)

    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    W -= lr * Xtrain.T.dot(y_train - Ytrain)
    b -= lr * (y_train - Ytrain).sum()
    if i % 1000 == 0:
        print("i, ctrain, ctest: ", i, ctrain, ctest)
    
print("Final train class rate: ", classification_rate(Ytrain, np.round(y_train)))
print("Final test class rate: ", classification_rate(Ytest, np.round(y_test)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()