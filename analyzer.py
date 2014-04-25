from nltk.corpus import cmudict
from nltk import pos_tag, word_tokenize, sent_tokenize


def tag_text(text):
    sentences = sent_tokenize(text)
    return map(lambda s: pos_tag(word_tokenize(s)), sentences)


def num_syllables(word, UNIQUE=True):
    """
    Returns the number of syllables in a word.

    Arguments:
    word -- a single word to be broken into syllables
    UNIQUE -- some words have multiple phonetic representations. If UNIQUE is
              True, the first such representation is used and an integer is
              returned. If UNIQUE is False, a list of syallabic counts is
              returned, one for each representation.
    """
    d = cmudict.dict()
    nsyls = [len(list(y for y in x if y[-1].isdigit()))
             for x in d[word.lower()]]

    if UNIQUE:
        return nsyls[0]
    else:
        return nsyls
