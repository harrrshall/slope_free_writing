#!/usr/bin/env python3
"""Crux judge collect (Claude-judge path). Reads the judge agents' raw score items from
/tmp/crux_judge/raw_scores.json (a flat list of {aid, overall, craft, coherence, imagery}, repeated K
times per aid), maps anon ids back to real passage_ids via mapping.json, averages the K realizations, and
writes results/features/judge_scores.parquet. Fails loudly if any passage is unscored (coverage guard).
"""
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "/tmp/crux_judge"
DIMS = ["overall", "craft", "coherence", "imagery"]
OUT = os.path.join(REPO, "results", "features", "judge_scores.parquet")


def main():
    mapping = json.load(open(os.path.join(D, "mapping.json")))
    raw = json.load(open(os.path.join(D, "raw_scores.json")))
    by_aid = {}
    for it in raw:
        rec = {d: float(it[d]) for d in DIMS}
        for d, v in rec.items():
            if not (1 <= v <= 100):
                raise ValueError(f"{it.get('aid')} {d}={v} out of range")
        by_aid.setdefault(it["aid"], []).append(rec)

    missing = [a for a in mapping if a not in by_aid]
    assert not missing, f"[collect] {len(missing)} passages unscored: {missing[:10]}"

    rows = []
    for aid, pid in mapping.items():
        draws = by_aid[aid]
        row = {"passage_id": pid, "judge_k": len(draws),
               "judge_overall_std": float(np.std([dr["overall"] for dr in draws]))}
        for d in DIMS:
            row[f"judge_{d}"] = float(np.mean([dr[d] for dr in draws]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"[collect] wrote {OUT} n={len(df)} (k per passage: {df.judge_k.min()}-{df.judge_k.max()}); "
          f"mean overall std across draws={df.judge_overall_std.mean():.2f}")
    print(df[[f"judge_{d}" for d in DIMS]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
