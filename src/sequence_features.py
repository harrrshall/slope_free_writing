#!/usr/bin/env python3
"""Order-aware (sequence) prosody features — EXTRACTION_UPGRADE_PLAN Stage 1.

The 14 summary stats in features_lib (sl_var, stress_runlen_var, ...) are ORDER-BLIND:
long-short-long-short and long-long-short-short have identical variance but read nothing
alike. These 10 features read the SAME ordered series (SL/SPW/SYMS/PHG) and preserve order,
to test whether order carries within-human quality signal the histograms miss.

FROZEN spec (design panel + synthesis, workflow wn6y05p9j, pre-registered in EXPERIMENT_LOG
2026-06-08). Every feature is:
  - order-sensitive (changes when the series is shuffled — the whole point),
  - scale-invariant (rank/autocorr/ratio/rate based: identical under series*c and (for ac/mk/
    turning) series+c) so it cannot repackage mean sentence/word length (M5),
  - finite-guarded (degenerate default 0.0 on empty/short/constant/zero-variance series),
  - M3-clean: a metronome / perfectly periodic series lands at an EXTREME, never the "good"
    region (alternation -> ac1 ~ -1; runs-z -> +6; turning-rate -> 1.0; bvr/lre -> floor).

Pure numpy. No scipy, no spaCy, no network. Series are built in features_lib.ordered_series.
"""
import hashlib
import numpy as np

ORDER_COLS = [
    "sl_ac1", "sl_mk", "sl_bvr", "sl_lre", "sl_turning",
    "spw_ac1", "syms_ac1", "syms_runz", "phg_ac1", "phg_lre",
]


# ---- generic order-aware primitives ----
def _ac1(series):
    """Lag-1 autocorrelation in [-1, 1]. Alternation -> ~-1, persistence -> +, noise -> ~0."""
    x = np.asarray(series, dtype=float)
    if len(x) < 3:
        return 0.0
    xc = x - x.mean()
    ss = float(xc @ xc)
    if ss == 0.0:
        return 0.0
    return float(xc[:-1] @ xc[1:] / ss)


def _mk(series):
    """Normalized Mann-Kendall S in [-1, 1]: signed direction of the arc (rising/falling/none).
    Rank-based -> invariant to any monotone rescale. Metronome and noise -> ~0 (M3 clean)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 4:
        return 0.0
    S = 0.0
    for i in range(n):
        S += np.sign(x[i + 1:] - x[i]).sum()
    return float(S / (n * (n - 1) / 2.0))


def _bvr(series):
    """Between-thirds / within-thirds variance ratio: paragraph-scale directional architecture.
    Metronome -> 0.0 (block means equal => floor, M3 clean). Scale-invariant (ratio)."""
    x = np.asarray(series, dtype=float)
    if len(x) < 6:
        return 0.0
    blocks = np.array_split(x, 3)
    gm = x.mean()
    between = sum(len(b) * (b.mean() - gm) ** 2 for b in blocks) / len(x)
    within = float(np.mean([b.var() for b in blocks]))
    if within == 0.0:
        return 0.0
    return float(between / within)


def _lre(series):
    """Low-frequency (slow) variance fraction, window=3: the text-only CWT-style multiscale probe.
    Metronome -> ~0.11 (high-freq power killed by the smoother), noise -> ~0.49, smooth arc -> ~0.87;
    both monotony and noise sit below a real arc (M3 clean). Scale-invariant (variance ratio)."""
    x = np.asarray(series, dtype=float)
    if len(x) < 6 or x.var() == 0.0:
        return 0.0
    sm = np.convolve(x, np.ones(3) / 3.0, mode="valid")
    return float(sm.var() / x.var())


def _turning(series):
    """Wallis-Moore turning-point rate in [0, 1]: oscillation density of the contour.
    M3 ANCHOR: metronome -> 1.0 (every interior point a strict extremum = MAXIMAL), so a high
    value is NEVER 'good'; read jointly with ac1/mk (metronome at the negative/zero extreme)."""
    x = np.asarray(series, dtype=float)
    if len(x) < 3:
        return 0.0
    c = 0
    for i in range(1, len(x) - 1):
        if (x[i] > x[i - 1] and x[i] > x[i + 1]) or (x[i] < x[i - 1] and x[i] < x[i + 1]):
            c += 1
    return float(c / (len(x) - 2))


def _runz(series):
    """Wald-Wolfowitz runs-test z for a BINARY series (SYMS): signed clustering vs alternation
    relative to the exact iid null. Clustering -> negative z, strict alternation -> +6 (extreme,
    not 'good'), iid -> ~0 by construction. Clipped to [-6, 6] so always finite."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 6:
        return 0.0
    n1 = int((x == 1).sum())
    n0 = int((x == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.0
    runs = 1 + int((x[1:] != x[:-1]).sum())
    mu = 1.0 + 2.0 * n1 * n0 / n
    var = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n * n * (n - 1))
    if var <= 0:
        return 0.0
    z = (runs - mu) / np.sqrt(var)
    return float(np.clip(z, -6.0, 6.0))


# ---- the 10 frozen features over the 4 series ----
def order_features(SL, SPW, SYMS, PHG):
    """Return the 10 order-aware feature columns from the four ordered series."""
    return {
        "sl_ac1": _ac1(SL),
        "sl_mk": _mk(SL),
        "sl_bvr": _bvr(SL),
        "sl_lre": _lre(SL),
        "sl_turning": _turning(SL),
        "spw_ac1": _ac1(SPW),
        "syms_ac1": _ac1(SYMS),
        "syms_runz": _runz(SYMS),
        "phg_ac1": _ac1(PHG),
        "phg_lre": _lre(PHG),
    }


def passage_seed(passage_id):
    """Deterministic per-passage seed for the shuffle control. md5, NOT python hash() (which is
    salted per process and would make the order-shuffle control non-reproducible)."""
    return int(hashlib.md5(str(passage_id).encode()).hexdigest()[:8], 16)


def order_features_shuffled(SL, SPW, SYMS, PHG, seed):
    """PRIMARY-2 control: independently permute each series with a frozen seed, then recompute.
    A genuine order feature MUST change; an order-invariant residue will not. If the augmented
    classifier does not beat this shuffled version, the 'order' claim dies (pre-registered)."""
    rng = np.random.default_rng(seed)
    return order_features(
        rng.permutation(np.asarray(SL, dtype=float)),
        rng.permutation(np.asarray(SPW, dtype=float)),
        rng.permutation(np.asarray(SYMS, dtype=float)),
        rng.permutation(np.asarray(PHG, dtype=float)),
    )
