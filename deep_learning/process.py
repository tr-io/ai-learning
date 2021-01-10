import numpy as np
import pandas as pd

# all preprocessing

def get_data():
    df = pd.read_csv('../sample_data/ecommerce_data.csv')
    data = df.to_numpy() # numpy matrix

    # matrices are row, column

    X = data[:, :-1]
    Y = data[:, -1]

    # normalize first column
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape # samples, features, remember: want D << N (ideal)
    X2 = np.zeros((N, D+3)) # due to the data: 4 categorical values, doing one hot encoding
    X2[:, 0:(D-1)] = X[:, 0:(D-1)] # all of the D - 1 columns are the same

    # one-hot time
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1
    
    return X2, Y

def get_binary_data():
    X, Y = get_data() # collect data
    # filter it, only taking classes 0 and 1
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    return X2, Y2
