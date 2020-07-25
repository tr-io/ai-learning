import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []

for line in open('../sample_data/data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# turn them into np arrays
X = np.array(X)
Y = np.array(Y)

# plot to see what it looks like
#plt.scatter(X, Y)
#plt.show()

# apply equations we learned to calculate a and b
# y_i = a_xi + b
denom = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denom
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom

# calculated predicted Y
Yhat = a*X + b

# plot all
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculate R^2
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("r-squared is: ", r2)