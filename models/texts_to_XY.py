# Takes two text files and converts to an X,Y pair
# for use in evaluating classifiers
# X is made by extracting ngrams then converting to bag of words
# sparse vector representation
import sys
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from features.meter import meter_ngrams
from features.analyzer import syllable_ngrams


def stringify_ngrams(blocks):
    """
    blocks: [list of [lists of [ngrams]]]
    Converts ngrams of the form [[1,2,3], [4,5,6]] to ['123', '456']
    for use with a CountVectorizer
    """
    return [[''.join(map(str, ngram)) for ngram in block] for block in blocks]


def texts_to_BOW(t1, t2, npara=100, syl_ngram_len=4):
    """
    Converts given texts to feature space using bag of words on
    syllable and meter ngrams.  Returns X,Y feature, label pair
    """
    print("Splitting...")

    # Split text into paragraphs (double newline), remove any extra newlines
    PAR_SPLIT = r'(?:\r\n){2,}|\r{2,}|\n{2,}'
    blocks1 = [t.replace('\n', ' ') for t in re.split(PAR_SPLIT, t1)][:npara]
    blocks2 = [t.replace('\n', ' ') for t in re.split(PAR_SPLIT, t2)][:npara]

    print("Processing meter ngrams...")
    # Meter patterns per block
    meter1 = stringify_ngrams([meter_ngrams(text) for text in blocks1])
    meter2 = stringify_ngrams([meter_ngrams(text) for text in blocks2])

    print("Processing syllable ngrams...")
    # Syllable patterns per block
    syl1 = stringify_ngrams([syllable_ngrams(text, syl_ngram_len)
                            for text in blocks1])
    syl2 = stringify_ngrams([syllable_ngrams(text, syl_ngram_len)
                            for text in blocks2])

    # Combine features into single list of ngrams
    # ngrams should not overlap
    t1_features = [meter + syl for meter, syl in zip(meter1, syl1)]
    t2_features = [meter + syl for meter, syl in zip(meter2, syl2)]

    ngrams = t1_features + t2_features
    labels = ['A'] * len(t1_features) + ['B'] * len(t2_features)

    print("Vectorizing ngrams...")
    # Setting identity analyzer assumes that input is already in ngram form
    vect = CountVectorizer(analyzer=lambda x: x)

    X = vect.fit_transform(ngrams)
    Y = np.array(labels)

    return X, Y


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} file1 file2 output".format(sys.argv[0]))
        exit()

    f1 = sys.argv[1]
    f2 = sys.argv[2]
    output = sys.argv[3]

    try:
        with open(f1) as f:
            t1 = f.read()
        with open(f2) as f:
            t2 = f.read()
    except:
        print("Error reading files")
        exit()

    XY = texts_to_BOW(t1, t2)

    with open(output, 'wb') as f:
        pickle.dump(XY, f)
        print("Saved extracted X,Y to {}".format(output))
