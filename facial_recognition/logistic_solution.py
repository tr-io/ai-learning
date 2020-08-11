import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate

class LogisticModel(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-7, reg=0, epochs=120000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.w = cp.random.randn(D) / cp.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1

        for i in range(epochs):
            pY = self.forward(X)

            # gradient descent step
            self.w -= learning_rate * (X.T.dot(pY - Y) + reg * self.w)
            self.b -= learning_rate * ((pY - Y).sum() + reg * self.b)
            
            if i % 20 == 0:
                pYvalid = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, cp.around(pYvalid)) # cp.round just means threshold of 0.5 for classification
                print("i: ", i, " cost: ", c, " error: ", e)
                if e < best_validation_error:
                    best_validation_error = e
        
        print("best validation error: ", best_validation_error)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.w) + self.b)
    
    def predict(self, X):
        pY = self.forward(X)
        return cp.around(pY) # binary classification with threshold of 0.5

    def score(self, X, Y):
        prediction = self.predict(X)
        return (1 - error_rate(Y, prediction))

def main():
    X, Y = getBinaryData()
    
    X0 = X[Y == 0, :]
    X1 = X[Y == 1, :]
    X1 = cp.repeat(X1, 9, axis=0)
    X = cp.vstack([X0, X1])
    Y = cp.array([0]*len(X0) + [1]*len(X1))

    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    model.score(X, Y)

if __name__ == "__main__":
    main()