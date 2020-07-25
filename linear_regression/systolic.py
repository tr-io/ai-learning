# Data (X1, X2, X3)
# X1 = systolic blood pressure ( consider this Y, what we want to predict )
# X2 = age in years
# X3 = weight in pounds

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('../sample_data/mlr02.xls')
X = df.to_numpy()

plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

# plot data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # 3d scatterplot
ax.scatter(X[:,1], X[:,2], X[:,0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]

# need to 3 linear aggresions
# 1. only age as input
# 2. only weight as input
# 3. both as input
# 4. compare the r-squared
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = np.dot(X, w)

    # compute r-squared
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print("r2 for age only: ", get_r2(X2only, Y))
print("r2 for weight only: ", get_r2(X3only, Y))
print("r2 for both: ", get_r2(X, Y))