import re
import analyzer


def find_all_matching_ngrams(text, ngram, ngram_extractor, PUNC=True):
    if not ngram:
        return None

    n = len(ngram)
    word_ngrams = analyzer.word_ngrams(text, n, PUNC=PUNC)
    target_ngrams = ngram_extractor(text, n)
    print word_ngrams, target_ngrams
    return list(zip(*filter(lambda x: x[1] == ngram, zip(word_ngrams, target_ngrams)))[0])


def find_matches_for_pairs(pairs, ngram):
    n = len(ngram)
    split = zip(*pairs)
    word_ngrams = analyzer.to_ngrams(split[0], n)
    target_ngrams = analyzer.to_ngrams(split[1], n)
    return list(zip(*filter(lambda x: x[1] == ngram, zip(word_ngrams, target_ngrams)))[0])


def find_pos_ngram(text, ngram):
    pairs = analyzer.tag_text(text)
    return find_matches_for_pairs(pairs, ngram)


def find_syllable_ngram(text, ngram):
    return find_all_matching_ngrams(text, ngram, analyzer.syllable_ngrams)


def find_etymology_ngram(text, ngram):
    pairs = analyzer.etymology_with_words(text)
    return find_matches_for_pairs(pairs, ngram)


def sentence_ngrams(text, n):
    sents = []
    for s in re.split('(?<=[.!?,\(\)-;:]) +', text):
        sents.append(s[:-1])
        sents.append(s[-1])
    return analyzer.to_ngrams(sents, n)


def find_syllable_count_ngram(text, ngram):
    n = len(ngram)
    sent_grams = sentence_ngrams(text, n)
    target_ngrams = analyzer.syllable_count_ngrams(text, n)
    return list(zip(*filter(lambda x: x[1] == ngram, zip(sent_grams, target_ngrams)))[0])


def find_word_count_ngram(text, ngram):
    n = len(ngram)
    sent_grams = sentence_ngrams(text, n)
    target_ngrams = analyzer.word_count_ngrams(text, n)
    return list(zip(*filter(lambda x: x[1] == ngram, zip(sent_grams, target_ngrams)))[0])
