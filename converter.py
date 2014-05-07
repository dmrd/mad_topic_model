from features import analyzer
from collections import Counter
import argparse
import csv

ngram_index = 0
ngram_indices = {}


def get_ngram_index(word):
    global ngram_index
    global ngram_indices
    if not word in ngram_indices:
        ngram_indices[word] = ngram_index
        ngram_index += 1
    return ngram_indices[word]


author_index = 0
author_indices = {}


def get_author_index(author):
    global author_index
    global author_indices
    if not author in author_indices:
        author_indices[author] = author_index
        author_index += 1
    return author_indices[author]


def convert_to_slda(data):
    def to_slda(ngrams):
        counter = Counter(ngrams)
        s = str(len(counter))
        for (e, c) in counter.items():
            index = get_ngram_index(e)
            s += " " + str(index) + ":" + str(c)
        return s

    model_string = ""
    author_string = ""
    for (author, ngrams) in data:
        index = get_author_index(author)
        author_string += str(index) + "\n"
        model_string += to_slda(ngrams) + "\n"
    return model_string, author_string


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

    with open(args.input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, quotechar='"')
        corpus = [(row[0].strip(), row[1].strip()) for row in reader][:100]
        corpus = [(row[0], extractor(row[1])) for row in corpus]
        (model_string, author_string) = convert_to_slda(corpus)

        file(args.output_model_file, 'w').write(model_string)
        file(args.output_author_file, 'w').write(author_string)

        with open(args.output_ngram_indices, 'w+') as f:
            for ngram in sorted(ngram_indices.iterkeys(), key=lambda k: ngram_indices[k]):
                f.write(str(ngram) + "\n")

        with open(args.output_author_indices, 'w+') as f:
            for author in sorted(author_indices.iterkeys(), key=lambda k: author_indices[k]):
                f.write(author + "\n")
