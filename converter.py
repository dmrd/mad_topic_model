"""
    Converts a CSV file to input for the SLDA model.
    Writes to four different files:
        1. output_model_file: contains frequency counts for the ngrams
        2. output_author_file: contains author labels for the documents
        3. output_ngram_indices: maps ngrams to integer indices
        4. output_author_indices: maps authors to integer indices
    The convert writes to these documents in an online manner to avoid
    doing tons of data processing before producing results. With that in
    mind, it should be run with Python's -u (unbuffed output) option.
    Example usage:
    python -um converter data/quora.csv model.txt author.txt ngram_dict.txt author_dict.txt syllable 2

    Please pardon the verbosity and use of global variables. Just trying to get shit done.
"""

from features import analyzer
from collections import Counter
import argparse
import csv

ngram_index = 0
ngram_indices = {}


def get_ngram_index(word, filename):
    global ngram_index
    global ngram_indices
    if not word in ngram_indices:
        with open(filename, 'a+') as f:
            f.write(str(word) + "\n")
        ngram_indices[word] = ngram_index
        ngram_index += 1
    return ngram_indices[word]


author_index = 0
author_indices = {}


def get_author_index(author, filename):
    global author_index
    global author_indices
    if not author in author_indices:
        with open(filename, 'a+') as f:
            f.write(author + "\n")
        author_indices[author] = author_index
        author_index += 1
    return author_indices[author]


def convert_to_slda(data, extractor, model_file, author_file, model_dict, author_dict):
    def to_slda(ngrams):
        counter = Counter(ngrams)
        s = str(len(counter))
        for (e, c) in counter.items():
            index = get_ngram_index(e, model_dict)
            s += " " + str(index) + ":" + str(c)
        return s

    with open(model_file, 'w+') as fm:
        with open(author_file, 'w+') as fa:

            for (author, doc) in data:
                index = get_author_index(author, author_dict)

                ngrams = extractor(doc)

                fa.write(str(index) + "\n")
                fm.write(to_slda(ngrams) + "\n")


ngram_parsers = {
    'pos': analyzer.pos_ngrams,
    'etymology': analyzer.etymology_ngrams,
    'word': analyzer.word_ngrams,
    'word_count': analyzer.word_count_ngrams,
    'syllable': analyzer.syllable_ngrams,
    'syllable_count': analyzer.syllable_count_ngrams
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert CSV documents to ngram text files for SLDA model.')
    parser.add_argument('input_file', help='input filename (CSV)')
    parser.add_argument('output_model_file',
                        help='output filename for ngram frequency counts')
    parser.add_argument('output_author_file',
                        help='output filename for authors')
    parser.add_argument(
        'output_ngram_indices', help='output filename for ngram indices')
    parser.add_argument(
        'output_author_indices', help='output filename for author indices')
    parser.add_argument(
        'ngram_type', help='type of ngram to extract, one of: pos, etymology, syllable, syllable_count, word, word_count (see analyzer for details)')
    parser.add_argument('n', type=int, help='n for the n-grams')
    args = parser.parse_args()

    if not args.ngram_type in ngram_parsers:
        raise ValueError(str(args.ngram_type) + " is not a valid ngram type.")
    extractor = lambda doc: ngram_parsers[args.ngram_type](doc, args.n)

    # wipe files, if existing
    open(args.output_ngram_indices, 'w+')
    open(args.output_author_indices, 'w+')

    with open(args.input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, quotechar='"')
        corpus = [(row[0].strip(), row[1].strip()) for row in reader]
        convert_to_slda(corpus, extractor, args.output_model_file,
                        args.output_author_file, args.output_ngram_indices, args.output_author_indices)
