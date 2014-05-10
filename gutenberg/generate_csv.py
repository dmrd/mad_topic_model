"""
    Generates a CSV with author, excerpt columns
"""

import os
import csv

DATA_FOLDER = "data"

# number of excerpts per author
EXCERPTS_PER_AUTHOR = 10

# excerpt size in lines
EXCERPT_SIZE = 30

# only look at files with the author's name and nothing else
file_names = [x for x in os.listdir(DATA_FOLDER) if ".txt" in x and "_small" not in x]

# list of tuples (author, excerpt)
results = []

for file_name in file_names:
    
    author = file_name.replace(".txt","")

    lines = [x.replace("\n", "") for x in open(DATA_FOLDER+"/"+file_name, "r").readlines()]

    for i in range(EXCERPTS_PER_AUTHOR):
        results.append((author, " ".join(lines[i*EXCERPT_SIZE:(i+1)*EXCERPT_SIZE])))

# write to one CSV file
writer = csv.writer(open("gutenberg.csv",'w'))
writer.writerows(results)