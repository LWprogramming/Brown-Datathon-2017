import ipdb
import numpy as np
from sklearn import svm # going to start with svm to try out.
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = np.load('cleaned_data.npy')
    data = data[:1000, :]
    labels = data[:, 0]
    features = data[:, 1:]

    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=1).fit(feature_train, label_train)

    # ipdb.set_trace()  ######### Break Point ###########

    print(clf.score(feature_test, label_test))
