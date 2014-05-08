"""
Generates SLDA models for all the CSV files
"""

import os
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

CSV_FILENAMES = {"nweekly" : "data/nweekly.csv", "quora" : "data/quora.csv"}
OUTPUT_FOLDER_NAME = "slda_input_files/"
NGRAM_PARSERS = ['pos','etymology','word','word_count','syllable','syllable_count','meter']
N_VALUES = [2, 3, 4, 5]

for filename in CSV_FILENAMES:
    for parser in NGRAM_PARSERS:
        for N in N_VALUES:
            prefix = "%s%s_%s_%s" % (OUTPUT_FOLDER_NAME,filename,parser,N)
            command = "python -um converter %s %s_model.txt %s_author.txt %s_ngram_dict.txt %s_author_dict.txt %s %s" % (CSV_FILENAMES[filename], prefix, prefix, prefix, prefix, parser, N)

            logging.info("Executing: %s" % command)
            os.system(command)