import unittest

from pyhealth.text_stats import tokenize, top_words, word_counts
from pyhealth.utils import chunked


class TestTextStats(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Patient's BP is 120/80."), ["patient's", "bp", "is", "120", "80"])

    def test_word_counts(self):
        self.assertEqual(word_counts("pain pain fever"), {"pain": 2, "fever": 1})

    def test_top_words(self):
        self.assertEqual(top_words("b a b c a b", k=2), ["b", "a"])


class TestChunked(unittest.TestCase):
    def test_chunked(self):
        self.assertEqual(chunked(range(5), 2), [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()
