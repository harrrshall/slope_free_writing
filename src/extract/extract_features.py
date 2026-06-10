#!/usr/bin/env python3
"""Extract per-passage features into features_<tier>.parquet.

Tier A = cheap (pronouncing + spaCy). Tier B = A + prosodic metrical-tension (slow,
only for the single permitted rescue round). Asserts no NaN/inf before writing.
"""
import argparse, os
import numpy as np
import pandas as pd
import spacy
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ root on path for library imports
from features_lib import extract_features, PROSODY_COLS, TIER_B_COLS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    rows = []
    for i, r in man.iterrows():
        path = os.path.join(REPO, "data", "passages", r["class"], r["passage_id"] + ".txt")
        with open(path) as f:
            text = f.read()
        feats = extract_features(text, nlp, tier=args.tier)
        feats.update(passage_id=r["passage_id"], **{"class": r["class"]},
                     source_id=r["source_id"], band=r["band"], word_count=r["word_count"])
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"[features] {i+1}/{len(man)}", flush=True)

    df = pd.DataFrame(rows)
    cols = PROSODY_COLS + (TIER_B_COLS if args.tier == "B" else [])
    feat_block = df[cols + ["mattr"]].to_numpy(dtype=float)
    if not np.isfinite(feat_block).all():
        bad = df.loc[~np.isfinite(df[cols + ["mattr"]]).all(axis=1), "passage_id"].tolist()
        raise SystemExit(f"[features] FATAL NaN/inf in: {bad[:10]}")

    out = os.path.join(REPO, args.out or f"results/features/features_{args.tier}.parquet")
    df.to_parquet(out, index=False)
    print(f"[features] wrote {out}  shape={df.shape}  cols={cols}")
    print("[features] per-class means (main band):")
    print(df[df.band == "main"].groupby("class")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
