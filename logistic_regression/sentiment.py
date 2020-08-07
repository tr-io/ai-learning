import nltk
import re
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.utils import shuffle
import heapq

"""
Helper methods
"""
def sigmoid(a):
    return 1 / (1 + cp.exp(-a))

def forward(X, W):
    return sigmoid(X.dot(W))

def cross_entropy(T, Y):
    return -cp.mean(T * cp.log(Y) + (1-T) * cp.log(1-Y))

"""
Utility methods
"""
def parse_train_data(path):
    """
    Parse data from the directory
    Return matrix of positive and negative reviews
    Only look at review_text key

    Features are one hot encoding bag of words
    """
    print("Setting up train data!")
    if path[-1] == '/':
        path = path[:-1]

    # open up the files and read them as xml
    # only look at review_text tags    
    positive_file = open(path + "/positive.review").read()
    positive_reviews = BeautifulSoup(positive_file, 'lxml')

    negative_file = open(path + "/negative.review").read()
    negative_reviews = BeautifulSoup(negative_file, 'lxml')

    pos_texts = positive_reviews.find_all('review_text')
    neg_texts = positive_reviews.find_all('review_text')

    # create bag of words model
    # 1. remove punctuation, special characters, and extra spaces
    pos_text = ""
    neg_text = ""
    for review in pos_texts:
        pos_text += review.get_text()
    
    for review in neg_texts:
        neg_text += review.get_text()

    pos_corpus = nltk.sent_tokenize(pos_text)
    neg_corpus = nltk.sent_tokenize(pos_text)

    for i in range(len(pos_corpus)):
        pos_corpus[i] = pos_corpus[i].lower()
        pos_corpus[i] = re.sub(r'\W',' ', pos_corpus[i])
        pos_corpus[i] = re.sub(r'\s+',' ', pos_corpus[i])

    for i in range(len(neg_corpus)):
        neg_corpus[i] = neg_corpus[i].lower()
        neg_corpus[i] = re.sub(r'\W',' ', neg_corpus[i])
        neg_corpus[i] = re.sub(r'\s+',' ', neg_corpus[i])
    
    # 2. tokenize sentences and create a dictionary that contains words and frequencies
    # O(n^2)
    word_freq = {}
    for s in pos_corpus:
        words = nltk.word_tokenize(s)
        for w in words:
            if w not in word_freq:
                word_freq[w] = 1
            else:
                word_freq[w] += 1

    # O(n^2)
    for s in neg_corpus:
        words = nltk.word_tokenize(s)
        for w in words:
            if w not in word_freq:
                word_freq[w] = 1
            else:
                word_freq[w] += 1
            
    word_freq = heapq.nlargest(200, word_freq, key=word_freq.get)

    # 3. convert each review_text to vector representation form
    # make sure to add the label
    review_vectors = [] # double array, convert to matrix

    # convert positive reviews to vector representation
    # append 1 if the word exists in word_freq else append 0
    # O(n^2)
    for s in pos_corpus:
        s_tokens = nltk.word_tokenize(s)
        sent_vec = []
        for token in word_freq:
            if token in s_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sent_vec.append(1) # append 1 at the end for the positive label
        review_vectors.append(sent_vec)
    
    # convert negative reviews to vector representation
    # O(n^2)
    for s in neg_corpus:
        s_tokens = nltk.word_tokenize(s)
        sent_vec = []
        for token in word_freq:
            if token in s_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sent_vec.append(0) # append 0 at the end for the negative label
        review_vectors.append(sent_vec)

    review_vectors = cp.asarray(review_vectors) # convert to matrix
    print("Done parsing data!")
    return review_vectors

def train_and_test(train_data):
    """
    Train weights
    """
    print("Learning weights!")
    N = train_data.shape[0]
    X = train_data[:, :-1] # features matrix
    T = train_data[:, -1] # targets matrix
    X, T = shuffle(X, T)

    # encode bias term into weights
    ones = cp.array([[1]*N]).T
    Xb = cp.concatenate((ones, X), axis=1)

    D = X.shape[1] # get number of features
    W = cp.random.randn(D + 1) / cp.sqrt(D + 1) # initialize random weights

    z = Xb.dot(W) # model output
    Y = sigmoid(z) # output of logistic

    costs = []
    lr = 0.0001 # learning rate for gradient descent
    l2 = 0.1 # lambda value for l1 regularization

    for i in range(30000):
        Y = forward(Xb, W)
        delta = Y - T
        W = W - lr * (Xb.T.dot(delta) - l2 * W)
        cost = cross_entropy(T, Y)
        if i % 1000 == 0:
            print("i, cost: ", i, cost)

        costs.append(cost)

    return W, costs

def main():
    # parse the data
    uns_train_data = parse_train_data("../sample_data/electronics")  # unshuffled training data
    # train model
    weights, costs = train_and_test(uns_train_data) 
    print("final weights: ", weights)
    plt.plot(costs, label='costs')
    plt.show()


if __name__ == "__main__":
    main()
