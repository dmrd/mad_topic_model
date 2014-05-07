from nltk import PorterStemmer, WordNetLemmatizer
from Levenshtein import ratio


class EtymologyDict(object):
    d = {}
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def load(self):
        complete_dict = file("origin/output.txt").readlines()
        for line in complete_dict:
            split = line.split(',')
            self.d[split[0].strip()] = split[1].strip()

    def lookup(self, s):
        if not self.d:
            self.load()

        try:
            return self.d[s.lower()]
        except KeyError:
            try:
                return self.d[self.stemmer.stem(s).lower()]
            except KeyError:
                try:
                    return self.d[self.lemmatizer.lemmatize(s).lower()]
                except KeyError:
                    (score, match) = max((ratio(s, t), t) for t in self.d)
                    self.d[s] = self.d[match]
                    return self.d[match]
