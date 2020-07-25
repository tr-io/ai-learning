import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5) * 10 # random data

true_w = np.array([1, 0.5, -0.5] + [0] * (D-3))
Y = X.dot(true_w) + np.random.randn(N)*0.5 #gaussian random noise

# do gradient descent
costs = []
w = np.random.randn(D) / np.sqrt(D)
lr = 0.001

l1 = 10.0 # lambda hyperparameter

for i in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - lr * (X.T.dot(delta) + l1*np.sign(w))

    mse = delta.dot(delta) / N 
    costs.append(mse)

plt.plot(costs)
plt.show()

print("final w: ", w)

plt.plot(true_w, label='true_w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()