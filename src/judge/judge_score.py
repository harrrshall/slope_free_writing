#!/usr/bin/env python3
"""Crux test, step 1 - score every great/modgreat/modhuman passage's prose quality with a FRONTIER LLM judge
(Gemini 2.5 Pro), BLIND, multi-realization, cached. Pre-reg + amendment: EXPERIMENT_LOG 2026-06-09.

Anti-cheating safeguards (audit-hardened):
- The prompt NEVER mentions rhythm / cadence / prosody / meter / stress (no contamination of the thing tested).
- The judge sees ONLY the passage text - no class label, source, author, era, or human-vs-AI hint.
- 1-100 scale across 4 dims (overall/craft/coherence/imagery) -> a STRONG, high-resolution baseline so prosody
  faces the highest bar.
- K independent realizations (temp 0.5) averaged (M15: never rest on a single LLM draw); per-passage overall
  std stored as a stability check. Cached by passage_id so re-runs cost nothing.

Usage:
  .venv/bin/python3 src/judge/judge_score.py            # score all (idempotent)
  .venv/bin/python3 src/judge/judge_score.py --limit 2  # smoke before full spend
Writes results/features/judge_scores.parquet.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import google.generativeai as genai

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "features", "judge_scores.parquet")
MODEL = "models/gemini-2.5-pro"
CLASSES = ["great", "modgreat", "modhuman"]
DIMS = ["overall", "craft", "coherence", "imagery"]
K = 2          # realizations per passage (M15)
TEMP = 0.5

PROMPT = (
    "You are an expert literary editor assessing the quality of a passage of English prose.\n"
    "Read the passage and rate it on four independent dimensions, each on an integer 1-100 scale "
    "(1 = very poor, 100 = exceptional, masterful):\n"
    "  overall   - overall writing quality\n"
    "  craft     - sentence-level skill, word choice, control\n"
    "  coherence - clarity and logical flow of ideas\n"
    "  imagery   - vividness and concreteness of description\n"
    "Judge ONLY the quality of the writing. Do not consider length or subject matter. "
    'Respond with ONLY a JSON object, e.g. {"overall": 72, "craft": 65, "coherence": 80, "imagery": 55}.\n\n'
    'PASSAGE:\n"""\n{TEXT}\n"""'
)


def _model():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    return genai.GenerativeModel(
        MODEL, generation_config={"temperature": TEMP, "response_mime_type": "application/json"})


def _parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    obj = json.loads(m.group(0) if m else txt)
    out = {d: float(obj[d]) for d in DIMS}
    for k, v in out.items():
        if not (1 <= v <= 100):
            raise ValueError(f"{k}={v} out of range")
    return out


def score_one(model, text, retries=4):
    """K realizations -> per-dim mean + overall std (stability)."""
    draws = []
    for _ in range(K):
        for i in range(retries):
            try:
                r = model.generate_content(PROMPT.replace("{TEXT}", text))
                draws.append(_parse(r.text)); break
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(2 * (i + 1))
    rec = {f"judge_{d}": float(np.mean([dr[d] for dr in draws])) for d in DIMS}
    rec["judge_overall_std"] = float(np.std([dr["overall"] for dr in draws]))
    rec["judge_k"] = len(draws)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N (smoke test)")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    m = man[(man.band == "main") & (man["class"].isin(CLASSES))].reset_index(drop=True)

    done = {}
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = {r.passage_id: dict(r) for _, r in prev.iterrows()}
        print(f"[judge] resuming: {len(done)} already scored")

    todo = [r for _, r in m.iterrows() if r["passage_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[judge] scoring {len(todo)} passages x K={K} realizations with {MODEL} (blind, 1-100, temp={TEMP})")

    model = _model()
    rows = [dict(passage_id=pid, **{k: done[pid][k] for k in done[pid] if k.startswith("judge_")})
            for pid in done]
    for j, r in enumerate(todo):
        pid, cls = r["passage_id"], r["class"]
        with open(os.path.join(REPO, "data", "passages", cls, pid + ".txt")) as f:
            text = f.read().strip()
        rows.append(dict(passage_id=pid, **score_one(model, text)))
        if (j + 1) % 10 == 0 or j + 1 == len(todo):
            print(f"[judge] {j+1}/{len(todo)} last={pid}", flush=True)
            pd.DataFrame(rows).to_parquet(OUT, index=False)   # checkpoint
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"[judge] wrote {OUT} n={len(df)}; mean overall std across draws={df['judge_overall_std'].mean():.2f}")
    print(df[[f"judge_{d}" for d in DIMS]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
