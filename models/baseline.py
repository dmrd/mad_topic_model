"""
Trains and tests simple logistic regression on ngram counts

Reads in data in same format as the slda model does
"""

import sys
import numpy as np
from sklearn.cross_validation import LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

prefix = sys.argv[1]
n_types = int(sys.argv[2])  # 0.5
nfolds = 10
if len(sys.argv) > 3:
    nfolds = int(sys.argv[3])


print("Loading data...")
with open(prefix + "_labels") as f:
    labels = [int(x.strip()) for x in f.readlines()]


# Files that need loading
data = [prefix + "_" + str(x) for x in range(n_types)]

# Read in files
lines = []
# List of ngram types
# ngram_types[i] is a list of length #docs
# each doc is a dictionary mapping ids to counts
ngram_types = []
ngram_max = []
for filename in data:
    f = open(filename, 'r')
    doc_ngrams = []
    for line in f:
        ngrams = {}
        for ngram in line.split()[1:]:
            id, count = ngram.split(':')
            ngrams[int(id)] = count
        doc_ngrams.append(ngrams)
    ngram_max.append(max([max(doc.keys()) for doc in doc_ngrams]))
    ngram_types.append(doc_ngrams)

# Convert into feature vector form
print("Vectorizing data...")
Xs = []
for i, ntype in enumerate(ngram_types):
    x = np.zeros((len(ntype), ngram_max[i] + 1))
    for j, doc in enumerate(ntype):
        for k in doc:
            x[j][k] = doc[k]
    Xs.append(x)

# Combine into a single feature array
X = np.hstack(Xs)
Y = np.array(labels)

print("Running cross validation...")
# Do actual classification
predictions = Y.copy()
clf = LogisticRegression()
# for train, test in LeaveOneOut(len(Y)):
for i, (train, test) in enumerate(KFold(len(Y), n_folds=nfolds)):
    print("Fold {}".format(i + 1))
    clf.fit(X[train], Y[train])
    predictions[test] = clf.predict(X[test])

print("Accuracy: {}".format(metrics.accuracy_score(Y, predictions)))

# print(metrics.classification_report(Y, predictions))
# print(metrics.confusion_matrix(Y, predictions))
