import sys
import json
from features import analyzer, meter

text_key = "name"
child_key = "children"

ngram_parsers = {
    'pos': analyzer.pos_ngrams,
    'etymology': analyzer.etymology_ngrams,
    'word_count': analyzer.word_count_ngrams,
    'syllable': analyzer.syllable_ngrams,
    'syllable_count': analyzer.syllable_count_ngrams,
    'meter': lambda x, y: meter.meter_ngrams(x)
}


def get_ngrams(text, n):
    data = {text_key: text, child_key: []}
    for ngram_type in ngram_parsers:
        parse = ngram_parsers[ngram_type]
        items, ngrams = parse(text, n, BODY=True)
        sub_data = {text_key: items, child_key: ngrams}
        data[child_key].append(sub_data)

if __name__ == "__main__":
    text = sys.stdin.readlines()
    print json.dumps(get_ngrams(text, 2))
