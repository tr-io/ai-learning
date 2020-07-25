import numpy as np
import matplotlib.pyplot as plt


# setting up & generate data
N = 50

X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# solve for the best weights
# w = (lambda * I + X.T dot X)^(-1) * X.T dot Y
X = np.vstack([np.ones(N), X]).T

# max likelihood - standard LR
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml)
plt.show()

# l2 solution (MAP)
l2 = 1000.0 # l2 penalty
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)

plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml, label='maximum likelihood')
plt.scatter(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()