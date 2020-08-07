import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# calculate cross entropy error

# center first 50 around (-2, -2)
# center last 50 samples around (+2, +2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# targets array
T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize weights
w = np.random.randn(D + 1)

# model output
z = Xb.dot(w)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

Y = sigmoid(z) # output of logistic

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    
    return E

# learn the weights w/ gradient descent
lr = 0.1 # learning rate
l2 = 0.1 # l2 lambda
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))
    
    w += lr * (np.dot((T-Y).T, Xb) - l2 * w) #L2 regularization
    Y = sigmoid(Xb.dot(w))

print("Final w: ", w)

# do L2 regularization

# use closed form solution
"""
w = np.array([0, 4, 4]) # bias is 0, (m_1 - m_2) = (2 - (-2), 2 - (-2))

z = Xb.dot(w)
Y = sigmoid(z)

print (cross_entropy(T, Y))

plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
"""