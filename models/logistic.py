# Uses leave one out cross validation with logistic regression to
# test classification accuracy for given file.
# Input file is produced by texts_to_XY.py and is a pickled tuple of
# features and labels (X, Y)
import pickle
import sys
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} input.pickle".format(sys.argv[0]))
        exit()

    with open(sys.argv[1], 'rb') as f:
        X, Y = pickle.load(f)

    predictions = Y.copy()
    clf = LogisticRegression()
    for train, test in LeaveOneOut(len(Y)):
        clf.fit(X[train], Y[train])
        predictions[test] = clf.predict(X[test])

    print(metrics.classification_report(Y, predictions))
    print(metrics.confusion_matrix(Y, predictions))
