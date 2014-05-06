import unittest
import features.analyzer as analyzer
import features.extract as extracter


class ExtracterTests(unittest.TestCase):

    def setUp(self):
        self.text = "This is preeminently the time to speak the truth, the whole truth, frankly and boldly. Nor need we shrink from honestly facing conditions in our country today. This great Nation will endure, as it has endured, will revive and will prosper."

    def test_find_pos_ngram(self):
        matches = extracter.find_pos_ngram(self.text, ('JJ', 'NN'))
        self.assertEqual(matches, [('whole', 'truth'), ('great', 'Nation')])

    def test_find_syllable_ngram(self):
        matches = extracter.find_syllable_ngram(self.text, (2, 1, 2))
        self.assertEqual(matches, [('frankly', 'and', 'boldly'),
                         ('Nation', 'will', 'endure'), ('endured', 'will', 'revive')])


class AnalyzerTests(unittest.TestCase):

    def test_fuzzy_lookup(self):
        self.assertEqual(analyzer.cmu_lookup("Dohan"),
                         ['D', 'R', 'OW1', 'AH0', 'N'])

    def test_syllables(self):
        def confirm(s, n):
            self.assertEqual(analyzer.num_syllables(s), n)

        confirm("dramatization", 5)
        confirm("hibachi", 3)
        confirm("ready", 2)

    def test_stress(self):
        self.assertEqual(
            analyzer.stress("fire"), [1, 0])

    def test_syllable_counts(self):
        self.assertEqual(analyzer.syllable_counts("Okay; I am ready."),
                         [([2], ';'), ([1, 1, 2], '.')])
        self.assertEqual(
            analyzer.syllable_counts("Okay; I am ready.", TOTAL=True),
            [(2, ';'), (4, '.')])

    def test_word_counts(self):
        self.assertEqual(analyzer.word_counts("Okay; I am ready. Go"),
                         [(1, ';'), (3, '.'), (1, None)])

    def test_stress_counts_by_syllable(self):
        self.assertEqual(
            analyzer.stress_counts_by_syllable("Okay; I am ready."),
            [([2, 1], ';'), ([1, 1, 1, 0], '.')])

    def test_ngrams(self):
        self.assertEqual(
            analyzer.syllable_ngrams("It is: overwhelming.", 2), [(1, 1), (1, 4)])
        self.assertEqual(
            analyzer.syllable_count_ngrams(
                "It is: overwhelming. We should go.", 2),
            [(2, 4), (4, 3)])


if __name__ == "__main__":
    unittest.main()
