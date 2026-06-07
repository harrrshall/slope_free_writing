#!/usr/bin/env python3
"""Extract lexical-baseline features into a parquet keyed by passage_id (joined to prosody later)."""
import argparse, os
import numpy as np
import pandas as pd
import spacy
from lexical_features import lexical_features, LEXICAL_COLS

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lexical_A.parquet")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(REPO, "manifest.csv"))
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    rows = []
    for i, r in man.iterrows():
        path = os.path.join(REPO, "data", "passages", r["class"], r["passage_id"] + ".txt")
        with open(path) as f:
            text = f.read()
        feats = lexical_features(text, nlp)
        feats.update(passage_id=r["passage_id"], **{"class": r["class"]},
                     source_id=r["source_id"], band=r["band"])
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"[lexical] {i+1}/{len(man)}", flush=True)

    df = pd.DataFrame(rows)
    block = df[LEXICAL_COLS].to_numpy(float)
    if not np.isfinite(block).all():
        bad = df.loc[~np.isfinite(df[LEXICAL_COLS]).all(axis=1), "passage_id"].tolist()
        raise SystemExit(f"[lexical] FATAL NaN/inf in: {bad[:10]}")
    out = os.path.join(REPO, args.out)
    df.to_parquet(out, index=False)
    print(f"[lexical] wrote {out} shape={df.shape}")
    print(df[df.band == "main"].groupby("class")[LEXICAL_COLS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
