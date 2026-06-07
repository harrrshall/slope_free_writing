#!/usr/bin/env python3
"""M3 guard (the one code-level enforcement that we reward VARIATION, not REGULARITY).

A true metronome (literal alternating stress '01'*N) must score ~0 on every variation
feature, and varied real prose must score STRICTLY HIGHER on each. We assert the relative
ordering (not guessed absolute thresholds). Run with: .venv/bin/python -m pytest test_m3_guard.py -q
"""
import itertools
import numpy as np
import pytest

VARIED = (
    "The old clock ticked. Somewhere, far beyond the shuttered windows and the long, "
    "cold corridor that the servants never used after dark, a door swung open, then shut, "
    "softly. She waited. Nothing. Then everything at once, a rush of footsteps, a cry, the "
    "shattering of glass against the flagstones, and after that a silence so complete it "
    "seemed to press upon the ears like deep water on a diver going down, and down, and down."
)


def metronome_variation_feats(n=120):
    """Variation features computed directly on a literal metronome '01'*n stress sequence."""
    seq = [int(c) for c in ("01" * n)]
    stressed = [i for i, s in enumerate(seq) if s == 1]
    gaps = np.diff(stressed)
    runs = [len(list(g)) for _, g in itertools.groupby(seq)]
    return dict(
        gap_cv=float(gaps.std() / gaps.mean()),
        gap_var=float(gaps.var()),
        stress_runlen_var=float(np.var(runs)),
    )


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def test_metronome_variation_is_zero():
    m = metronome_variation_feats()
    assert m["gap_cv"] < 1e-9
    assert m["gap_var"] < 1e-9
    assert m["stress_runlen_var"] < 1e-9


def test_varied_prose_beats_metronome(nlp):
    from features_lib import extract_features
    prose = extract_features(VARIED, nlp, tier="A")
    metro = metronome_variation_feats()
    for k in ("gap_cv", "gap_var", "stress_runlen_var"):
        assert prose[k] > metro[k], f"{k}: prose {prose[k]:.4f} !> metronome {metro[k]:.4f}"
    # sentence-length variation: the metronome has none; varied prose must show some.
    assert prose["sl_var"] > 0.0
    assert prose["sl_cv"] > 0.0
