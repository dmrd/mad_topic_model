import sys
import json
import argparse
from features import analyzer, meter

body_key = "body"
child_key = "children"
type_key = "type"

ngram_parsers = {
    'pos': analyzer.pos_ngrams,
    'etymology': analyzer.etymology_ngrams,
    'word': analyzer.word_ngrams,
    #'word_count': analyzer.word_count_ngrams,
    'syllable': analyzer.syllable_ngrams,
    #'syllable_count': analyzer.syllable_count_ngrams,
    #'meter': lambda x, y, BODY=True: meter.meter_ngrams(x)
}


def prettify(ngram):
    ngram = str(ngram)
    ngram = ngram.replace(',', ', ')
    return ngram


def get_ngrams(text, n):
    data = {body_key: text, child_key: [], type_key: "text"}
    for ngram_type in ngram_parsers:
        parse = ngram_parsers[ngram_type]
        items, ngrams = parse(text, n, BODY=True)
        ngrams = [{body_key: prettify(ngram), child_key: [], type_key: ngram_type}
                  for ngram in ngrams]
        sub_data = {body_key: prettify(items), child_key: ngrams, type_key: ngram_type}
        data[child_key].append(sub_data)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract all ngrams and convert to JSON output.')
    parser.add_argument('n', type=int, help='ngram length')
    args = parser.parse_args()
    text = ''.join(sys.stdin.readlines())
    data = get_ngrams(text, args.n)
    print json.dumps(data)
