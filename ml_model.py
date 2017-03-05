# import ipdb
import numpy as np
# from sklearn import svm # going to start with svm to try out.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

'''
Output:
Gaussian Naive Bayes
Training metrics:
accuracy score: 0.692408
Cross-validation metrics:
accuracy_score: 0.608632

Multinomial Naive Bayes
Training metrics:
accuracy score: 0.78425
Cross-validation metrics:
accuracy_score: 0.784233

Logistic regression
Training metrics:
accuracy score: 0.784659
Cross-validation metrics:
accuracy_score: 0.78466

Analysis: looks like GaussianNB is performing a lot worse than the other two; the other two seem to be underfitting
'''

def print_training_metrics(clf, features, labels):
    '''
    assuming clf is already trained.
    '''
    predictions = clf.predict(features)
    print('accuracy score: ' + '%.3f' % accuracy_score(labels, predictions))
    # print('precision score: ' + '%.3f' % precision_score(labels, predictions))
    # print('recall score: ' + '%.3f' % recall_score(labels, predictions))
    # print('F1 score: ' + '%.3f' % f1_score(labels, predictions))

def print_cv_metrics(clf, features, labels):
    '''
    cv score
    '''
    print('accuracy_score: ' + '%.3f' % np.mean(cross_val_score(clf, features, labels, cv=5, scoring='accuracy')))
    # print('precision score: ' + '%.3f' % np.mean(cross_val_score(clf, features, labels, cv=5, scoring='precision')))
    # print('recall score: ' + '%.3f' % np.mean(cross_val_score(clf, features, labels, cv=5, scoring='recall')))
    # print('F1 score: ' + '%.3f' % np.mean(cross_val_score(clf, features, labels, cv=5, scoring='f1')))

if __name__ == '__main__':
    data = np.load('cleaned_data.npy')
    # data = data[:1000, :]
    labels = data[:, 0]
    features = data[:, 1:]

    # cv is about 0.69
    gnb = GaussianNB()
    gnb.fit(features, labels)
    print('Gaussian Naive Bayes')
    print('Training metrics:')
    print_training_metrics(gnb, features, labels) # training error
    print('Cross-validation metrics:')
    print_cv_metrics(GaussianNB(), features, labels) # this is technically separate because I'm passing in a fresh classifier so it's still like keeping the cv-data separate from the training effectively
    print('')

    # # cv is about 0.68
    mnb = MultinomialNB()
    mnb.fit(features, labels)
    print('Multinomial Naive Bayes')
    print('Training metrics:')
    print_training_metrics(mnb, features, labels) # training error
    print('Cross-validation metrics:')
    print_cv_metrics(MultinomialNB(), features, labels) # this is technically separate because I'm passing in a fresh classifier so it's still like keeping the cv-data separate from the training effectively
    print('')

    # cv is about 0.78 (TAKES A LONG TIME TO TRAIN)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf.fit(features, labels)
    print('Logistic regression')
    print('Training metrics:')
    print_training_metrics(clf, features, labels) # training error
    print('Cross-validation metrics:')
    print_cv_metrics(LogisticRegression(solver='lbfgs', multi_class='multinomial'), features, labels) # this is technically separate because I'm passing in a fresh classifier so it's still like keeping the cv-data separate from the training effectively
    print('')
