#!/usr/bin/env python3
"""Extract the augmented Stage-1 feature parquet: 14 summary prosody + mattr + 10 REAL order
features + 10 SHUFFLED order features (PRIMARY-2 control), one aligned row per passage.

Parses each passage ONCE and feeds the SAME ordered series to both the summary stats and the
order features (features_lib._stress_series / _syntactic_series), so order and histogram features
cannot disagree about the underlying signal. The shuffled columns use a frozen md5(passage_id)
seed so the order-shuffle control is exactly reproducible. Asserts all features finite before write.
"""
import os
import numpy as np
import pandas as pd
import spacy

from features_lib import (stress_features, syntactic_features, mattr,
                          _stress_series, _syntactic_series, PROSODY_COLS)
from sequence_features import (order_features, order_features_shuffled, passage_seed, ORDER_COLS)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER_SHUF_COLS = [c + "_shuf" for c in ORDER_COLS]


def main():
    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    rows = []
    for i, r in man.iterrows():
        path = os.path.join(REPO, "data", "passages", r["class"], r["passage_id"] + ".txt")
        with open(path) as f:
            text = f.read()
        doc = nlp(text)
        words = [t.text for t in doc if t.is_alpha]
        syms, spw, _ = _stress_series(words)
        sl, phg = _syntactic_series(doc)

        feats = {}
        feats.update(stress_features(words))           # 14 summary prosody (PROSODY_COLS) ...
        feats.update(syntactic_features(doc))
        feats["mattr"] = mattr(doc)
        feats.update(order_features(sl, spw, syms, phg))                                   # REAL order
        seed = passage_seed(r["passage_id"])
        feats.update({k + "_shuf": v for k, v in
                      order_features_shuffled(sl, spw, syms, phg, seed).items()})          # SHUFFLED order
        feats.update(passage_id=r["passage_id"], **{"class": r["class"]},
                     source_id=r["source_id"], band=r["band"], word_count=r["word_count"],
                     n_sent=len(sl))
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"[seq] {i+1}/{len(man)}", flush=True)

    df = pd.DataFrame(rows)
    check_cols = PROSODY_COLS + ["mattr"] + ORDER_COLS + ORDER_SHUF_COLS
    block = df[check_cols].to_numpy(dtype=float)
    if not np.isfinite(block).all():
        bad = df.loc[~np.isfinite(df[check_cols]).all(axis=1), "passage_id"].tolist()
        raise SystemExit(f"[seq] FATAL NaN/inf in: {bad[:10]}")

    out = os.path.join(REPO, "results", "features", "features_seq_A.parquet")
    df.to_parquet(out, index=False)
    print(f"[seq] wrote {out}  shape={df.shape}")
    print(f"[seq] order cols: {ORDER_COLS}")
    # quick per-class peek at the order features (main band)
    print(df[df.band == "main"].groupby("class")[ORDER_COLS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
