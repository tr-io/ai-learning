import numpy as np
import matplotlib.pyplot as plt

# load data
X = []
Y = []

for line in open('../sample_data/data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    # do this just like multiple LR
    # have your extra constant term
    X.append([1, x, x*x])
    Y.append(float(y))

# convert to np array
X = np.array(X)
Y = np.array(Y)

# plot data
plt.scatter(X[:,1], Y)
plt.show()

# calculate weights
# same thing as MLR - only difference is how we created input table X
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot all together
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()

# compute r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("r-squared: ", r2)