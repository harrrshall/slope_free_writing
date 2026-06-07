#!/usr/bin/env python3
"""Shared text->prosody feature extraction (Tier A cheap; Tier B adds prosodic).

All features are derived FROM TEXT only (CMUdict stress via pronouncing + spaCy
segmentation; prosodic uses eSpeak G2P for OOV, never audio) -> honors M6/D4.

Feature groups (used by the rhythm-in-top-3 gate, criterion 3):
  stress_timing : gap_mean, gap_var, gap_cv, stress_runlen_var, stress_bin_entropy
  sentence_rhythm: sl_var, sl_cv
  clause_spacing: ph_mean, ph_var
  syllable_dist : spw_mean, spw_var, spw_skew
  confound      : sl_mean, oov_rate          (NOT a rhythm group)
  metrical_tension (Tier B): mt_mean, mt_var
'mattr' is a lexical covariate/baseline, NOT part of the prosody matrix.

The VARIATION features (var/cv/entropy/run-length) are what must score HIGHER on
varied prose than on a metronome (M3); the *_mean / oov_rate are regularity/confound.
"""
import itertools
import numpy as np
from scipy import stats
import pronouncing

PROSODY_COLS = [
    "stress_bin_entropy", "stress_runlen_var", "gap_mean", "gap_var", "gap_cv",
    "sl_mean", "sl_var", "sl_cv", "ph_mean", "ph_var",
    "spw_mean", "spw_var", "spw_skew", "oov_rate",
]
TIER_B_COLS = ["mt_mean", "mt_var"]

FEATURE_GROUPS = {
    "stress_timing": ["gap_mean", "gap_var", "gap_cv", "stress_runlen_var", "stress_bin_entropy"],
    "sentence_rhythm": ["sl_var", "sl_cv"],
    "clause_spacing": ["ph_mean", "ph_var"],
    "syllable_dist": ["spw_mean", "spw_var", "spw_skew"],
    "confound": ["sl_mean", "oov_rate"],
    "metrical_tension": ["mt_mean", "mt_var"],
}
RHYTHM_GROUPS = ["stress_timing", "sentence_rhythm", "clause_spacing", "syllable_dist", "metrical_tension"]

_PUNCT_BOUNDARY = {",", ";", ":", "—", "–", "-", "(", ")"}


def _entropy(counts):
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def stress_features(words):
    """Binary stress sequence over syllables -> timing features + spw + oov_rate."""
    syms, spw, oov, tot = [], [], 0, 0
    for w in words:
        tot += 1
        ph = pronouncing.phones_for_word(w.lower())
        if not ph:
            oov += 1
            continue
        st = pronouncing.stresses(ph[0])
        if not st:
            continue
        spw.append(len(st))
        syms.extend(1 if d in "12" else 0 for d in st)
    oov_rate = oov / tot if tot else 0.0

    stressed = [i for i, s in enumerate(syms) if s == 1]
    gaps = np.diff(stressed) if len(stressed) >= 2 else np.array([])
    gap_mean = float(gaps.mean()) if gaps.size else 0.0
    gap_var = float(gaps.var()) if gaps.size else 0.0
    gap_cv = float(gaps.std() / gaps.mean()) if (gaps.size and gaps.mean() > 0) else 0.0
    runs = [len(list(g)) for _, g in itertools.groupby(syms)] if syms else []
    stress_runlen_var = float(np.var(runs)) if runs else 0.0
    if gaps.size:
        _, counts = np.unique(gaps, return_counts=True)
        stress_bin_entropy = _entropy(counts)
    else:
        stress_bin_entropy = 0.0

    spw_mean = float(np.mean(spw)) if spw else 0.0
    spw_var = float(np.var(spw)) if spw else 0.0
    # scipy.stats.skew is NaN on a constant array -> guard on var>0 AND n>2 (M-fix)
    spw_skew = float(stats.skew(spw)) if (len(spw) > 2 and np.var(spw) > 0) else 0.0

    return dict(stress_bin_entropy=stress_bin_entropy, stress_runlen_var=stress_runlen_var,
                gap_mean=gap_mean, gap_var=gap_var, gap_cv=gap_cv,
                spw_mean=spw_mean, spw_var=spw_var, spw_skew=spw_skew, oov_rate=oov_rate)


def syntactic_features(doc):
    """Sentence-length and clause/phrase-boundary spacing from a spaCy Doc."""
    sl = [sum(1 for t in s if t.is_alpha) for s in doc.sents]
    sl = [x for x in sl if x > 0]
    sl_arr = np.array(sl, dtype=float)
    sl_mean = float(sl_arr.mean()) if sl_arr.size else 0.0
    sl_var = float(sl_arr.var()) if sl_arr.size else 0.0
    sl_cv = float(sl_arr.std() / sl_arr.mean()) if (sl_arr.size and sl_arr.mean() > 0) else 0.0

    phg, gap = [], 0
    for t in doc:
        if t.is_alpha:
            gap += 1
        is_boundary = (t.is_punct and t.text in _PUNCT_BOUNDARY) or t.is_sent_end
        if is_boundary:
            phg.append(gap)
            gap = 0
    phg = [g for g in phg if g > 0]
    ph_arr = np.array(phg, dtype=float)
    ph_mean = float(ph_arr.mean()) if ph_arr.size else 0.0
    ph_var = float(ph_arr.var()) if ph_arr.size else 0.0

    return dict(sl_mean=sl_mean, sl_var=sl_var, sl_cv=sl_cv, ph_mean=ph_mean, ph_var=ph_var)


def mattr(doc, window=50):
    toks = [t.text.lower() for t in doc if t.is_alpha]
    if not toks:
        return 0.0
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    ttrs = [len(set(toks[i:i + window])) / window for i in range(len(toks) - window + 1)]
    return float(np.mean(ttrs))


def metrical_tension(doc, nlp_prosodic):
    """Tier B: per-sentence prosodic metrical-tension proxy (slow). Text-only G2P."""
    import prosodic
    scores = []
    for s in doc.sents:
        txt = s.text.strip()
        if len(txt.split()) < 3:
            continue
        try:
            t = prosodic.Text(txt)
            t.parse()
            if not t.lines:
                continue
            bp = t.lines[0].best_parse
            if bp is None or not bp.num_sylls:
                continue
            scores.append(float(bp.score) / float(bp.num_sylls))  # tension per syllable
        except Exception:
            continue
    arr = np.array(scores, dtype=float)
    mt_mean = float(arr.mean()) if arr.size else 0.0
    mt_var = float(arr.var()) if arr.size else 0.0
    return dict(mt_mean=mt_mean, mt_var=mt_var)


def extract_features(text, nlp, tier="A"):
    doc = nlp(text)
    words = [t.text for t in doc if t.is_alpha]
    feats = {}
    feats.update(stress_features(words))
    feats.update(syntactic_features(doc))
    feats["mattr"] = mattr(doc)
    if tier == "B":
        feats.update(metrical_tension(doc, nlp))
    return feats
