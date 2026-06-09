#!/usr/bin/env python3
"""Guards for the order-aware (sequence) features. Enforces the pre-registered properties:
finiteness, scale-invariance (M5: cannot repackage mean length), order-sensitivity (the whole
point), and M3 metronome behavior (a perfectly periodic series lands at an EXTREME, never 'good').
Run: .venv/bin/python -m pytest tests/test_sequence.py -q
"""
import math
import numpy as np
import pytest

from sequence_features import (_ac1, _mk, _bvr, _lre, _turning, _runz,
                               order_features, order_features_shuffled, passage_seed, ORDER_COLS)

SCALAR_FEATS = [_ac1, _mk, _bvr, _lre, _turning]            # defined on any real series
STRUCT = [3, 5, 8, 12, 18, 12, 8, 5, 3, 7, 15, 4, 9, 20, 6]  # an arc + tail, clearly ordered


# ---- finiteness on the degenerate battery ----
@pytest.mark.parametrize("series", [[], [5], [5, 5], [5, 5, 5], [7, 7, 7, 7, 7, 7], [3, 1, 4, 1, 5]])
def test_all_features_finite(series):
    feats = order_features(series, series, [s % 2 for s in series], series)
    for k, v in feats.items():
        assert math.isfinite(v), f"{k} not finite on {series}"


def test_constant_series_is_zero():
    c = [9] * 12
    feats = order_features(c, c, [1] * 12, c)
    for k, v in feats.items():
        assert v == 0.0, f"{k} should be 0.0 on a constant series, got {v}"


# ---- scale invariance (M5): cannot be a repackaged mean length ----
def test_scale_invariance_multiplicative():
    x = np.array(STRUCT, dtype=float)
    for f in SCALAR_FEATS:
        assert abs(f(x) - f(7.0 * x)) < 1e-9, f"{f.__name__} not invariant under x*7"


def test_shift_invariance_for_centered_features():
    x = np.array(STRUCT, dtype=float)
    for f in (_ac1, _mk, _turning, _bvr, _lre):
        assert abs(f(x) - f(x + 100.0)) < 1e-9, f"{f.__name__} not invariant under x+100"


# ---- order sensitivity: the value MUST move when the series is shuffled ----
def test_order_sensitive():
    x = np.array(STRUCT, dtype=float)
    xs = x[[7, 2, 14, 0, 9, 4, 11, 1, 13, 5, 8, 3, 12, 6, 10]]  # fixed permutation
    moved = [f for f in (_ac1, _mk, _turning, _bvr, _lre) if abs(f(x) - f(xs)) > 1e-6]
    assert len(moved) >= 4, f"order features barely move under shuffle: only {len(moved)}/5 changed"


# ---- M3: a metronome / perfectly periodic series lands at an EXTREME, never the 'good' region ----
def test_metronome_extremes():
    metro = [0, 1] * 60                       # perfect alternation, len 120
    assert _turning(metro) == 1.0             # MAXIMAL jaggedness (so high is never 'good')
    assert _ac1(metro) < -0.9                 # alternation -> ~-1 (anti-cadence extreme)
    assert _runz(metro) == 6.0                # excess-alternation z saturates the +6 clip
    assert abs(_bvr(metro)) < 1e-9            # block means equal -> floor 0
    assert _lre(metro) < 0.2                  # all power at high frequency -> low slow-fraction


def test_mk_metronome_and_ramp():
    assert abs(_mk([0, 1] * 60)) < 0.05       # no net trend -> ~0 (M3 clean, not rewarded)
    assert _mk(list(range(1, 30))) == 1.0     # strict rising ramp -> +1
    assert _mk(list(range(30, 1, -1))) == -1.0


# ---- shuffle control is exactly reproducible (frozen md5 seed) ----
def test_shuffle_reproducible():
    seed = passage_seed("64317")
    a = order_features_shuffled(STRUCT, STRUCT, [s % 2 for s in STRUCT], STRUCT, seed)
    b = order_features_shuffled(STRUCT, STRUCT, [s % 2 for s in STRUCT], STRUCT, seed)
    assert a == b, "md5-seeded shuffle must be deterministic"
    assert set(a.keys()) == set(ORDER_COLS)
