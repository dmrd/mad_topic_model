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
import random

prefix = sys.argv[1]
n_types = int(sys.argv[2])  # 0.5
fraction_train = float(sys.argv[3])  # Float in 0-1.0

# Files that need loading
files = [prefix + "_labels"] + [prefix + "_" + str(x) for x in range(n_types)]

# Read in files
lines = []
for filename in files:
    f = open(filename, 'r')
    lines.append(f.readlines())

# Make sure all input has same # lines
assert(all(len(x) == len(lines[0]) for x in lines))

# Generate permutation and shuffle lines
permutation = list(range(len(lines[0])))
random.shuffle(permutation)

shuffled = []
for doc in lines:
    shuffled.append([doc[i] for i in permutation])

train_size = int(len(lines[0]) * fraction_train)

for path, doc in zip(files, shuffled):
    base, name = os.path.split(path)
    with open(os.path.join(base, "test_" + name), 'w') as f:
        f.write(''.join(doc[:train_size]))
    with open(os.path.join(base, "train_" + name), 'w') as f:
        f.write(''.join(doc[train_size:]))
