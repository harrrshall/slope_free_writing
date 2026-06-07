#!/usr/bin/env python3
"""Lexical-slop baseline features (the H2 "what existing slop detectors already see" channel).

Frozen spec D12. Lists loaded ONLY from vendored data/slop_lists/ (pinned slop-forensics SHA
c313f04); no runtime network. All densities are per-1000 alpha tokens (length-invariant). n-gram
matching mirrors slop-forensics: NFKC+lower, strip NLTK stopwords, then form content n-grams.
mattr lives HERE (lexical), never in PROSODY_COLS (D11) -> never on both sides of the H2 increment.
"""
import os, re, json, unicodedata
from collections import Counter
import numpy as np
import pronouncing
from features_lib import mattr as _mattr, PROSODY_COLS

REPO = os.path.dirname(os.path.abspath(__file__))
SLOP_DIR = os.path.join(REPO, "data", "slop_lists")

LEXICAL_COLS = [
    "slopword_density", "slopbigram_density", "sloptrigram_density", "not_x_but_y_rate",
    "distinct2", "distinct3", "rep_topk", "mattr", "mean_word_len", "fk_grade",
]
# distinct2/distinct3/rep_topk partially proxy VARIATION (M3-adjacent); evaluate.py runs the
# residual test BOTH ways (full lexical vs lexical-minus-distinct-n) as a pre-registered diagnostic.
DISTINCT_COLS = ["distinct2", "distinct3", "rep_topk"]

# Phase 1b (corrected, see EXPERIMENT_LOG + M12): the diversity-proxy features circularly remove
# prosodic variation, so the binding orthogonality baseline is SLOP-ONLY (genuine slop lexicon +
# readability). DIVERSITY is reported separately, never in the slop baseline.
DIVERSITY_COLS = ["distinct2", "distinct3", "rep_topk", "mattr"]
SLOP_BASELINE_COLS = ["slopword_density", "slopbigram_density", "sloptrigram_density",
                      "not_x_but_y_rate", "mean_word_len", "fk_grade"]

NOT_X_BUT_Y = re.compile(r"(?i)not [^.!?]{3,60} but ")   # antislop slop_regexes.txt line 1


def _load():
    sw = set(open(os.path.join(SLOP_DIR, "nltk_stopwords.txt")).read().split())
    words = {x[0].lower() for x in json.load(open(os.path.join(SLOP_DIR, "slop_list.json")))}
    bi = {x[0].lower() for x in json.load(open(os.path.join(SLOP_DIR, "slop_list_bigrams.json")))}
    tri = {x[0].lower() for x in json.load(open(os.path.join(SLOP_DIR, "slop_list_trigrams.json")))}
    return sw, words, bi, tri


STOP, SLOPWORDS, SLOPBI, SLOPTRI = _load()


def _tokens(text):
    norm = unicodedata.normalize("NFKC", text).lower()
    toks = re.findall(r"[a-z']+", norm)
    return [t.strip("'") for t in toks if t.strip("'")]


def _syllables(word):
    ph = pronouncing.phones_for_word(word)
    if ph:
        return max(1, sum(c.isdigit() for c in ph[0]))
    return max(1, len(re.findall(r"[aeiouy]+", word)))     # vowel-group fallback, text-only (M6)


def lexical_features(text, nlp):
    doc = nlp(text)
    toks = _tokens(text)
    n = len(toks) or 1
    sw_hits = sum(1 for t in toks if t in SLOPWORDS)

    content = [t for t in toks if t not in STOP]
    bigrams = [" ".join(content[i:i + 2]) for i in range(len(content) - 1)]
    trigrams = [" ".join(content[i:i + 3]) for i in range(len(content) - 2)]
    bi_hits = sum(1 for b in bigrams if b in SLOPBI)
    tri_hits = sum(1 for t in trigrams if t in SLOPTRI)

    allbi = [" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)]
    alltri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    distinct2 = len(set(allbi)) / len(allbi) if allbi else 0.0
    distinct3 = len(set(alltri)) / len(alltri) if alltri else 0.0
    rep_topk = sum(c for _, c in Counter(toks).most_common(10)) / n

    n_words = sum(1 for tk in doc if tk.is_alpha) or 1
    n_sents = len([s for s in doc.sents if any(tk.is_alpha for tk in s)]) or 1
    n_syll = sum(_syllables(tk.text.lower()) for tk in doc if tk.is_alpha)
    fk = 0.39 * (n_words / n_sents) + 11.8 * (n_syll / n_words) - 15.59

    return dict(
        slopword_density=1000.0 * sw_hits / n,
        slopbigram_density=1000.0 * bi_hits / n,
        sloptrigram_density=1000.0 * tri_hits / n,
        not_x_but_y_rate=1000.0 * len(NOT_X_BUT_Y.findall(text)) / n,
        distinct2=distinct2, distinct3=distinct3, rep_topk=rep_topk,
        mattr=_mattr(doc),
        mean_word_len=float(np.mean([len(t) for t in toks])) if toks else 0.0,
        fk_grade=fk,
    )


assert not (set(LEXICAL_COLS) & set(PROSODY_COLS)), "LEXICAL_COLS must be disjoint from PROSODY_COLS"
