#!/usr/bin/env python3
"""Lexical-feature sanity: n-gram matching is alive, and lexical is disjoint from prosody.

Run: .venv/bin/python -m pytest test_lexical.py -q
"""
import pytest
from features_lib import PROSODY_COLS

SLOPPY = ("She took a deep breath. Her voice was barely above a whisper, a kaleidoscope of "
          "emotion she could not help but feel as the silence stretched between them.")


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def test_lexical_disjoint_from_prosody():
    from lexical_features import LEXICAL_COLS
    assert not (set(LEXICAL_COLS) & set(PROSODY_COLS))


def test_ngram_matching_is_alive(nlp):
    # If stopword stripping or list loading were broken, these would silently be 0.
    from lexical_features import lexical_features
    f = lexical_features(SLOPPY, nlp)
    assert f["sloptrigram_density"] > 0, "trigram matching dead (stopwords/list problem)"
    assert f["slopbigram_density"] > 0, "bigram matching dead"
    assert f["slopword_density"] > 0


def test_features_finite(nlp):
    import math
    from lexical_features import lexical_features, LEXICAL_COLS
    f = lexical_features(SLOPPY, nlp)
    for c in LEXICAL_COLS:
        assert math.isfinite(f[c]), f"{c} not finite"
