"""Small helpers for summarizing free-text clinical notes."""

import re
from collections import Counter
from typing import Dict, List

_WORD = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens of ``text``."""
    return [w.lower() for w in _WORD.findall(text)]


def word_counts(text: str) -> Dict[str, int]:
    """Number of occurrences of each word in ``text``."""
    return dict(Counter(tokenize(text)))


def top_words(text: str, k: int = 5) -> List[str]:
    """The ``k`` most frequent words in ``text``, ties broken alphabetically."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    counts = word_counts(text)
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]


def type_token_ratio(text: str) -> float:
    """Distinct words divided by total words; 0.0 for empty text."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def sentence_lengths(text: str) -> List[int]:
    """Number of words in each sentence of ``text``."""
    sentences = re.split(r"[.!?]+", text)
    return [len(tokenize(s)) for s in sentences if tokenize(s)]
