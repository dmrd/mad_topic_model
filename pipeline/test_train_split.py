"""
Generates a test/train split of given files.

Shuffles:
    prefix_labels
    prefix_0 ... prefix_n
where n is n_types - 1

Prepends outputs with test_ and train_

This is hacky but works for what we need
"""

import sys
import os
import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
print(sys.argv)

prefix = sys.argv[1]
n_types = int(sys.argv[2])  # Number of ngram types (usually 6)
folds = int(sys.argv[3])  # Number of folds to use

# Files that need loading
files = [prefix + "_labels"] + [prefix + "_" + str(x) for x in range(n_types)]

# Read in files
lines = []
for filename in files:
    f = open(filename, 'r')
    lines.append(f.readlines())

# Make sure all input has same # lines
assert(all(len(x) == len(lines[0]) for x in lines))

labels = np.array(map(lambda x: int(x.strip()), lines[0]))

# for fid, (test, train) in enumerate(KFold(len(lines[0]), n_folds=folds,
#                                           shuffle=True)):
for fid, (test, train) in enumerate(StratifiedKFold(labels,
                                                    n_folds=folds)):
    test_fold = []
    train_fold = []
    for doc in lines:
        test_fold.append([doc[i] for i in test])
        train_fold.append([doc[i] for i in train])
    for path, testd, traind in zip(files, test_fold, train_fold):
        base, name = os.path.split(path)
        with open(os.path.join(base, "fold{}_test_{}".format(fid, name)), 'w') as f:
            f.write(''.join(testd))
        with open(os.path.join(base, "fold{}_train_{}".format(fid, name)), 'w') as f:
            f.write(''.join(traind))
