from nltk.corpus import cmudict
from nltk import pos_tag, word_tokenize, sent_tokenize


PUNCTUATION_TAGS = ['.', ':', ',']
d = cmudict.dict()


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


def syllabic_representation(text):
    """
    Maps text to a list of syllables and puncutation marks, e.g.:
    >>> syllabic_representation("Okay; I am ready.")
    >>> [2, ';', 1, 1, 2, '.']
    """
    def to_syllables((word, tag)):
        if tag in PUNCTUATION_TAGS:
            return word
        return num_syllables(word)
    return map(to_syllables, tag_text(text))


def syllable_counts(text):
    """
    Returns a list of (syllable_counts, punctuation) pairs, where
    syllable_counts is a list of integers representing the number of
    syllables in the words preceding the punctuation. If the text ends without
    a piece of punctuation, the final punctuation will be None. E.g.:
    >>> syllable_counts("Okay; I am ready.")
    >>> [([2], ';'), ([1, 1, 2], '.')]
    """
    result = []
    temp = []
    for x in syllabic_representation(text):
        if type(x) is int:
            temp.append(x)
        else:
            result.append((temp, x))
            temp = []
    if temp:
        result.append((temp, None))
    return result


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


def stress_counts(text, SYLLABLES=True):
    """
    Returns a list of (stress_counts, punctuation) pairs, where stress_counts
    is a list of integers and None values preceding the punctuation. If the
    text ends without punctuation, the final punctuation value will be None.
    E.g.:
    >>> stress_counts("Okay; I am ready.")
    >>> [([('OW', 2), ('K', 0), ('EY', 1)], ';'), ([('AY', 1), ('AE', 1), ('M', 0), ('R', 0), ('EH', 1), ('D', 0), ('IY', -1)], '.')]
    >>> stress_counts("Okay; I am ready.", SYLLABLES=False)
    >>> [([2, 0, 1], ';'), ([1, 1, 0, 0, 1, 0, -1], '.')]
    """
    result = []
    temp = []
    for (word, tag) in tag_text(text):
        if tag in PUNCTUATION_TAGS:
            result.append((temp, word))
            temp = []
        else:
            temp += stress(word, SYLLABLES=SYLLABLES)
    if temp:
        result.append((temp, None))
    return result


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
    nsyls = [len(list(y for y in x if y[-1].isdigit()))
             for x in d[word.lower()]]

    if UNIQUE:
        return nsyls[0]
    else:
        return nsyls


def stress(word, SYLLABLES=True):
    """
    Maps a word to its syllables and their stresses. (1) indicates primary,
    (2) secondary, and (-1) neutral, and (0) no stress. If SYLLABLES is False,
    only the stress information is returned. E.g.:
    >>> stress("fire")
    >>> [('F', 0), ('AY', 1), ('ER', -1)]
    >>> stress("fire", SYLLABLES=False)
    >>> [0, 1, -1]
    """
    def extract_stress(s):
        if s[-1].isdigit():
            val = int(s[-1])
            if val == 0:
                return (s[:-1], -1)
            else:
                return (s[:-1], val)
        return (s, 0)
    syllables = d[word.lower()][0]
    stresses = map(extract_stress, syllables)
    if not SYLLABLES:
        return map(lambda x: x[1], stresses)
    return stresses
