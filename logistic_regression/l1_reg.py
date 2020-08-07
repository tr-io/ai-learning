import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5) * 10 # uniformly distributed between -5 and 5
true_w = np.array([1, 0.5, -0.5] + [0] * (D-3)) # generate solution

Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N) * 0.5)) # generate the targets

costs = []
w = np.random.randn(D) / np.sqrt(D)
lr = 0.001
l1 = 10.0

for i in range(5000):
    yhat = sigmoid(X.dot(w))
    delta = yhat - Y
    w = w - lr * (X.T.dot(delta) + l1 * np.sign(w))

    cost = -(Y * np.log(yhat) + (1 - Y) * np.log(1 - yhat)).mean() + l1 * np.abs(w).mean()
    costs.append(cost)

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true w')
plt.plot(w, label = 'w map')
plt.legend()
plt.show()