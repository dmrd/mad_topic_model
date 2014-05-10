"""
Strip out documents that have any empty ngram types

This is hacky but works for what we need
"""

import sys
import os
import random

prefix = sys.argv[1]
n_types = int(sys.argv[2])  # 0.5

# Files that need loading
files = [prefix + "_labels"] + [prefix + "_" + str(x) for x in range(n_types)]

# Read in files
lines = []
for filename in files:
    f = open(filename, 'r')
    lines.append(f.readlines())

# Make sure all input has same # lines
assert(all(len(x) == len(lines[0]) for x in lines))

ignored = set()
# Find 0s in all but labels file
for doc in lines[1:]:
    for i, l in enumerate(doc):
        if l[0] == '0':
            ignored.add(i)

print ignored

for i, (path, doc) in enumerate(zip(files, lines)):
    base, name = os.path.split(path)
    with open(os.path.join(base, name + "_cleaned"), 'w') as f:
        f.write(''.join([l for j, l in enumerate(doc) if j not in ignored]))
