from nltk import PorterStemmer, WordNetLemmatizer
from Levenshtein import ratio


def etydict():
    d = {}
    complete_dict = file("origin/output.txt").readlines()
    for line in complete_dict:
        split = line.split(',')
        d[split[0].strip()] = split[1].strip()
    return d


d = etydict()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def lookup(s):
    try:
        return d[s.lower()]
    except KeyError:
        try:
            print "Stemming: " + stemmer.stem(s)
            return d[stemmer.stem(s).lower()]
        except KeyError:
            try:
                print "Lemmatizing: " + lemmatizer.lemmatize(s)
                return d[lemmatizer.lemmatize(s).lower()]
            except KeyError:
                (score, match) = max((ratio(s, t), t) for t in d)
                print "Levenshtein: " + match
                d[s] = d[match]
                return d[match]
