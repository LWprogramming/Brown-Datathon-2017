import numpy as np
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
import ipdb

if __name__ == '__main__':
    np.random.seed(7) # fix for reproducibility
    data = np.load('cleaned_data.npy')
    # data = data[:1000, :]
    labels = data[:, 0]
    features = data[:, 1:]

    # cv is about 0.69
    # gnb = GaussianNB()
    # scores = cross_val_score(gnb, features, labels, cv=5)
    # print(np.mean(scores))

    # cv is about 0.68
    # mnb = MultinomialNB()
    # scores = cross_val_score(mnb, features, labels, cv=5)
    # print(np.mean(scores))

    # cv is about 0.78 (TAKES A LONG TIME TO TRAIN)
    # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    # scores = cross_val_score(clf, features, labels, cv=5)
    # print(np.mean(scores))
