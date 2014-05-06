import analyzer


def find_matching_ngram(text, ngram, ngram_extractor):
    if not ngram:
        return None

    n = len(ngram)
    word_ngrams = analyzer.word_ngrams(text, n)
    target_ngrams = ngram_extractor(text, n)
    match = word_ngrams[target_ngrams.index(ngram)]
    return match


def find_all_matching_ngrams(text, ngram, ngram_extractor):
    if not ngram:
        return None

    n = len(ngram)
    word_ngrams = analyzer.word_ngrams(text, n)
    target_ngrams = ngram_extractor(text, n)
    return list(zip(*filter(lambda x: x[1] == ngram, zip(word_ngrams, target_ngrams)))[0])


def find_pos_ngram(text, ngram):
    return find_matching_ngram(text, ngram, analyzer.pos_ngrams)


def find_syllable_ngram(text, ngram):
    return find_matching_ngram(text, ngram, analyzer.syllable_ngrams)


def find_syllable_count_ngram(text, ngram):
    return find_matching_ngram(text, ngram, analyzer.syllable_count_ngrams)


def find_word_count_ngram(text, ngram):
    return find_matching_ngram(text, ngram, analyzer.word_count_ngrams)
