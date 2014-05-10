"""
Filter down dataset to authors with more than K documents

Reads in data in same format as the slda model does.
"""

import sys
import os
import numpy as np

prefix = sys.argv[1]
n_types = int(sys.argv[2])  # 0.5
K = 20
if len(sys.argv) > 3:
    K = int(sys.argv[3])


# Files that need loading
files = [prefix + "_labels"] + [prefix + "_" + str(x) for x in range(n_types)]

# Read in files
lines = []
for filename in files:
    f = open(filename, 'r')
    lines.append(f.readlines())

# Make sure all input has same # lines
assert(all(len(x) == len(lines[0]) for x in lines))


labels = map(lambda x: int(x.strip()), lines[0])
counts = np.bincount(labels)

# Write the ngram files
for i, (path, doc) in enumerate(zip(files[1:], lines[1:])):
    base, name = os.path.split(path)
    with open(os.path.join(base, name + "_k"), 'w') as f:
        for j, l in enumerate(doc):
            if counts[labels[j]] >= K:
                f.write(l)

# Write labels file, reassign labels to stay sequential

indices = (counts >= K).nonzero()[0]
new_map = {}
for i in range(len(indices)):
    new_map[indices[i]] = i
print("Authors with more than {} docs: {}".format(K, indices))
print("{} in total".format(len(indices)))

base, name = os.path.split(files[0])
with open(os.path.join(base, name + "_k"), 'w') as f:
    for label in labels:
        if counts[label] >= K:
            f.write("{}\n".format(new_map[label]))
