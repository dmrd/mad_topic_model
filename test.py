import unittest
import analyzer


class AnalyzerTests(unittest.TestCase):

    def test_syllables(self):
        def confirm(s, n):
            self.assertEqual(analyzer.num_syllables(s), n)

        confirm("dramatization", 5)
        confirm("hibachi", 3)
        confirm("ready", 2)

    def test_syllable_counts(self):
        self.assertEqual(analyzer.syllable_counts("Okay; I am ready."),
                         [([2], ';'), ([1, 1, 2], '.')])

    def test_word_counts(self):
        self.assertEqual(analyzer.word_counts("Okay; I am ready. Go"),
                         [(1, ';'), (3, '.'), (1, None)])


if __name__ == "__main__":
    unittest.main()
