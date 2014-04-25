import unittest
import analyzer


class AnalyzerTests(unittest.TestCase):

    def test_syllables(self):
        def confirm(s, n):
            self.assertEqual(analyzer.num_syllables(s), n)

        confirm("dramatization", 5)
        confirm("hibachi", 3)
        confirm("ready", 2)

    def test_stress(self):
        self.assertEqual(analyzer.stress("fire"),
                         [('F', 0), ('AY', 1), ('ER', -1)])
        self.assertEqual(
            analyzer.stress("fire", SYLLABLES=False), [0, 1, -1])

    def test_syllable_counts(self):
        self.assertEqual(analyzer.syllable_counts("Okay; I am ready."),
                         [([2], ';'), ([1, 1, 2], '.')])

    def test_word_counts(self):
        self.assertEqual(analyzer.word_counts("Okay; I am ready. Go"),
                         [(1, ';'), (3, '.'), (1, None)])

    def test_stress_counts(self):
        self.assertEqual(analyzer.stress_counts("Okay; I am ready."),
                         [([('OW', 2), ('K', 0), ('EY', 1)], ';'), ([('AY', 1), ('AE', 1), ('M', 0), ('R', 0), ('EH', 1), ('D', 0), ('IY', -1)], '.')])
        self.assertEqual(
            analyzer.stress_counts("Okay; I am ready.", SYLLABLES=False),
            [([2, 0, 1], ';'), ([1, 1, 0, 0, 1, 0, -1], '.')])


if __name__ == "__main__":
    unittest.main()
