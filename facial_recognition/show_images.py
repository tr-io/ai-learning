import numpy as np
import matplotlib.pyplot as plt

from util import getData

# array index corresponds to the value of the label
# string is the meaning
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    X, Y, Xtest, Ytest = getData()

    print(Y)

    while True:
        for i in range(7): # loop through labels
            x, y = X[Y == i], Y[Y == i] # select data that corresponds to this label
            print("x, y: ", x, y)
            N = len(y)
            print("N: ", N)
            j = np.random.choice(N) # select a random data point (index) (image)
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(label_map[y[j]])
            plt.show()
        prompt = input('Y to quit:')
        if prompt == 'Y':
            break

if __name__ == "__main__":
    main()
