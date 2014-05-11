import os
from nltk import PorterStemmer, WordNetLemmatizer
from Levenshtein import ratio


class EtymologyDict(object):
    d = {}
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def load(self):
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'origin/output.txt')
        complete_dict = file(filename).readlines()
        for line in complete_dict:
            split = line.split(',')
            self.d[split[0].strip()] = split[1].strip()

    def lookup(self, s):
        if not self.d:
            self.load()

        try:
            print "Using self"
            return self.d[s.lower()]
        except KeyError:
            try:
                print "Using stemmer: " + self.stemmer.stem(s).lower()
                return self.d[self.stemmer.stem(s).lower()]
            except KeyError:
                try:
                    print "Using lemmatizer: " + self.lemmatizer.lemmatize(s).lower()
                    return self.d[self.lemmatizer.lemmatize(s).lower()]
                except KeyError:
                    (score, match) = max((ratio(s, t), t) for t in self.d)
                    self.d[s] = self.d[match]
                    print "Using Levenshtein: " + match
                    return self.d[match]
