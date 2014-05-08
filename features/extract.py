import analyzer


def find_all_matching_ngrams(text, ngram, ngram_extractor, PUNC=True):
    if not ngram:
        return None

    n = len(ngram)
    word_ngrams = analyzer.word_ngrams(text, n, PUNC=True)
    target_ngrams = ngram_extractor(text, n)
    return list(zip(*filter(lambda x: x[1] == ngram, zip(word_ngrams, target_ngrams)))[0])


def find_pos_ngram(text, ngram):
    return find_all_matching_ngrams(text, ngram, analyzer.pos_ngrams)


def find_syllable_ngram(text, ngram):
    return find_all_matching_ngrams(text, ngram, analyzer.syllable_ngrams, PUNC=False)
