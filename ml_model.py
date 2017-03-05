import ipdb
import numpy as np
# from sklearn import svm # going to start with svm to try out.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
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
