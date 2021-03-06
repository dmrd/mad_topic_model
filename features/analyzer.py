from string import punctuation
from nltk.corpus import cmudict
from nltk.util import ngrams
from nltk import pos_tag, word_tokenize, sent_tokenize
from collections import Counter
from Levenshtein import ratio
from etymology_dict import EtymologyDict


TRUE_PUNCTUATION = ['!', '(', ')', ',', '-', '.', ':', ';', '?']
FALSE_PUNCTUATION = ['#', '$', '<', '=', '>', '+', '*', '@', '~', '^']
PUNCTUATION_TAGS = ['.', ':', ',']
d = cmudict.dict()
e = EtymologyDict()


def cmu_lookup(s, APPROX=True):
    """
    A wrapper around the CMU Dictionary lookup that uses approximate
    matching (via Levenshtein distance) for unknown words, when APPROX
    is True. E.g.:
    >>> cmu_lookup("Dohan")
    >>> ['D', 'R', 'OW1', 'AH0', 'N'] # Closest match: drohan
    """
    s = s.lower()
    try:
        return d[s][0]
    except KeyError:
        if APPROX:
            (score, match) = max((ratio(s, t), t) for t in d)
            d[s] = d[match]
            return d[match][0]
        raise

#
#  Utility methods
#


def to_ngrams(items, n, BODY=False):
    grams = ngrams(items, n)
    if BODY:
        return items, grams
    return grams


#
#  Part-of-Speech
#


def tag_sentences(text):
    """
    Splits the text into sentences and then tokenizes them individually.
    Returns a list of lists of (word, tag) pairs, where each sublist
    represents a sentence.
    """
    sentences = sent_tokenize(text)
    return map(lambda s: pos_tag(word_tokenize(s)), sentences)


def tag_text(text):
    """Tags a body of text, returning a list of (word, tag) pairs."""
    return sum(tag_sentences(text), [])


def pos_ngrams(text, n, BODY=False):
    """Extracts POS ngrams for a body of text."""
    pos_tags = [tag for (w, tag) in tag_text(text)]
    return to_ngrams(pos_tags, n, BODY=BODY)


def word_ngrams(text, n, PUNC=True, BODY=False):
    def filter(tag):
        return PUNC or not tag in PUNCTUATION_TAGS
    tokenized_text = [t for (t, tag) in tag_text(text) if filter(tag)]
    return to_ngrams(tokenized_text, n, BODY=BODY)


#
#  Syllables
#


def syllabic_representation(text):
    """
    Maps text to a list of syllables and puncutation marks, e.g.:
    >>> syllabic_representation("Okay; I am ready.")
    >>> [2, ';', 1, 1, 2, '.']
    """
    def to_syllables((word, tag)):
        if tag in PUNCTUATION_TAGS or word in TRUE_PUNCTUATION:
            return word
        return num_syllables(word)
    return map(to_syllables, tag_text(text))


def num_syllables(word):
    """
    Returns the number of syllables in a word.
    """
    return len(list(y for y in cmu_lookup(word) if y[-1].isdigit()))


def stress(word, SECONDARY=True):
    """
    Maps a word to its phones and their stresses. (1) indicates primary,
    (2) secondary, and (0) no stress. If SECONDARY is False, no distinction
    is made between types (1) and (2) stress. E.g.:
    >>> stress("fire")
    >>> [1, 0]
    """
    def extract_stress(s):
        n = int(s[-1])
        if not SECONDARY and n > 0:
            return 1
        return n
    syllables = filter(lambda x: x[-1].isdigit(), cmu_lookup(word))
    return map(extract_stress, syllables)


def syllable_counts(text, TOTAL=False):
    """
    Returns a list of (syllable_counts, punctuation) pairs, where
    syllable_counts is a list of integers representing the number of
    syllables in the words preceding the punctuation. If TOTAL is True,
    then the lists of syllables are summed. If the text ends without
    a piece of punctuation, the final punctuation will be None.
    E.g.:
    >>> syllable_counts("Okay; I am ready.")
    >>> [([2], ';'), ([1, 1, 2], '.')]
    >>> syllable_counts("Okay; I am ready.", TOTAL=True)
    >>> [(2, ';'), (4, '.')]
    """
    result = []

    def reset():
        if TOTAL:
            return 0
        return []
    temp = reset()

    for x in syllabic_representation(text):
        if type(x) is int:
            if not TOTAL:
                x = [x]
            temp += x
        else:
            result.append((temp, x))
            temp = reset()
    if temp:
        result.append((temp, None))
    return result


def syllable_ngrams(text, n, BODY=False):
    """
    Returns the n-grams of syllable usage by breaking each word down into
    (# syllables) and then taking n-grams on that sequence. In this way,
    it essentially ignores punctuation. E.g.:
    >>> syllable_ngrams("It is: overwhelming.", 2)
    >>> [(1, 1), (1, 4)]
    """
    syllables = syllabic_representation(text)
    return to_ngrams(syllables, n, BODY=BODY)


def syllable_count_ngrams(text, n, BODY=False):
    """
    Returns the n-grams of syllable counts between punctuation. That is,
    it sums (# syllables) for all the words between punctuation marks and
    returns the n-grams on that sequence. E.g.:
    >>> syllable_count_ngrams("It is: overwhelming. We should go.", 2)
    >>> [(2, 4), (4, 3)]
    """
    syllables = syllable_counts(text, TOTAL=True)
    syllables = list(sum(syllables, ()))
    return to_ngrams(syllables, n, BODY=BODY)


def word_counts(text):
    """
    Returns a list of (word_counts, punctuation) pairs, where word_counts
    is the number of words preceding the punctuation. If the text ends without
    a piece of punctuation, the final punctuation will be None. E.g.:
    >>> word_counts("Okay; I am ready.")
    >>> [(1, ';'), (3, '.')]
    """
    # Could use syllable_counts, return length of lists; faster to do manually
    result = []
    counter = 0
    for (word, tag) in tag_text(text):
        if tag in PUNCTUATION_TAGS:
            result.append((counter, word))
            counter = 0
        else:
            counter += 1
    if counter:
        result.append((counter, None))
    return result


def word_count_ngrams(text, n, BODY=False):
    """
    Returns the n-grams of word counts between punctuation.
    """
    counts = word_counts(text)
    counts = list(sum(counts, ()))
    return to_ngrams(counts, n, BODY=BODY)


def stress_counts_by_syllable(text, SECONDARY=True):
    """
    Returns a list of (stress_counts, punctuation) pairs, where stress
    is evaluated on a per-syllable basis, and stress_counts
    is a list of integers and None values preceding the punctuation. If the
    text ends without punctuation, the final punctuation value will be None.
    E.g.:
    >>> stress_counts_by_syllable("Okay; I am ready.")
    >>> [([2, 1], ';'), ([1, 1, 1, 0], '.')]
    """
    result = []
    temp = []
    for (word, tag) in tag_text(text):
        if tag in PUNCTUATION_TAGS:
            result.append((temp, word))
            temp = []
        else:
            temp += stress(word, SECONDARY=SECONDARY)
    if temp:
        result.append((temp, None))
    return result

#
#  Etymology
#


def etymology_with_words(text):
    exclude = set(punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = word_tokenize(text)
    return map(lambda w: (w, e.lookup(w)), text)


def etymology_representation(text):
    exclude = set(punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = word_tokenize(text)
    return map(e.lookup, text)


def etymology_ngrams(text, n, BODY=False):
    representation = etymology_representation(text)
    return to_ngrams(representation, n, BODY=BODY)


def etymology_frequencies(text, n):
    representation = etymology_representation(text)
    return Counter(representation)
