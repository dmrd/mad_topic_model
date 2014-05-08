"""
    This script cleans up the scrapy output CSV to be compliant with
    converter.py for this project

        python clean.py nweekly.csv

"""

import sys
import csv

# list of URLs already written
already_written = []

input_file = open(sys.argv[1], 'r')
output_file = open("output.csv", 'w')
reader = csv.reader(input_file)
writer = csv.writer(output_file)

for row in reader:
    body = row[0].strip()
    url = row[1].split("?")[0]
    author = row[4].strip()

    if url not in already_written:
        already_written.append(url)
        writer.writerow([author, body])