#!/usr/bin/env python3
"""EXPLORATORY probe (not a pre-registered binding test): Paul Graham essays as a HIGH-QUALITY MODERN
HUMAN pile vs the existing AI pile (slop) and the amateur-modern-human pile (modhuman).

PG fills a gap the project has lacked: an acknowledged-strong CONTEMPORARY human writer (breaks the
pre-1928 era confound of 'great'). HONEST CONFOUNDS (stated, not controlled): PG is non-fiction ESSAY
while slop/modhuman are FICTION, so PG-vs-slop conflates human-vs-AI with essay-vs-fiction (M5 genre).
Read as exploratory; the clean binding contrasts remain the pre-registered ones. Does NOT touch held-out.
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import balanced_accuracy_score

from features_lib import extract_features, PROSODY_COLS
from lexical_features import lexical_features, SLOP_BASELINE_COLS
from evaluate import logreg, _oof_ba, group_perm_test, group_bootstrap_increment_ci, group_bootstrap_ci

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)


def clean(text):
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)                       # footnote markers
    lines = text.splitlines()
    if lines and re.match(r"^(January|February|March|April|May|June|July|August|September|October|"
                          r"November|December)\s+\d{4}", lines[0].strip()):
        lines = lines[1:]                                            # leading date line
    text = "\n".join(lines)
    text = re.sub(r"(?is)Thanks to .{0,200}? for reading drafts.*$", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_essay(text, nlp, lo=220, hi=320):
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    passages, cur, cw, target = [], [], 0, int(rng.integers(lo, hi))
    for s in sents:
        cur.append(s); cw += len(s.split())
        if cw >= target:
            passages.append(" ".join(cur)); cur, cw, target = [], 0, int(rng.integers(lo, hi))
    if cw >= 150:
        passages.append(" ".join(cur))
    return passages


def build_pg(nlp):
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "data", "texts", "pg", "*.txt"))):
        slug = os.path.splitext(os.path.basename(path))[0]
        text = clean(open(path).read())
        for i, p in enumerate(chunk_essay(text, nlp)):
            feats = extract_features(p, nlp, tier="A")
            feats.update(lexical_features(p, nlp))
            feats.update(passage_id=f"pg_{slug}_{i}", **{"class": "pg"},
                         source_id=f"pg:{slug}", word_count=len(p.split()))
            rows.append(feats)
    return pd.DataFrame(rows)


def load_existing():
    of = pd.read_parquet(os.path.join(REPO, "results", "features", "features_A_p1c.parquet"))
    ol = pd.read_parquet(os.path.join(REPO, "results", "features", "lexical_A_p1c.parquet"))
    df = of.merge(ol[["passage_id"] + [c for c in SLOP_BASELINE_COLS if c in ol.columns]], on="passage_id")
    return df[(df.band == "main") & (df["class"].isin(["slop", "modhuman", "great"]))]


def contrast(dev, a, b, perm_n=1000):
    d = dev[dev["class"].isin([a, b])].reset_index(drop=True)
    y, g = d["class"].to_numpy(), d["source_id"].to_numpy()
    pros = [c for c in PROSODY_COLS if c in d.columns]
    slop = [c for c in SLOP_BASELINE_COLS if c in d.columns]
    oof_p, ba_p = _oof_ba(logreg(), d[pros].to_numpy(float), y, g)
    oof_s, ba_s = _oof_ba(logreg(), d[slop].to_numpy(float), y, g)
    oof_c, ba_c = _oof_ba(logreg(), d[pros + slop].to_numpy(float), y, g)
    _, ba_len = _oof_ba(logreg(), d[["sl_mean"]].to_numpy(float), y, g)
    ilo, ihi, imean = group_bootstrap_increment_ci(y, oof_c, oof_s, g)
    plo, phi = group_bootstrap_ci(y, oof_p, g)
    pt = group_perm_test(logreg(), d[pros].to_numpy(float), y, g, n=perm_n)
    return dict(contrast=f"{a}_vs_{b}", n=int(len(d)), n_groups=int(pd.unique(g).size),
                class_counts=d["class"].value_counts().to_dict(),
                prosody_only=ba_p, prosody_ci=[plo, phi], prosody_perm_p=pt["perm_p"],
                slop_only=ba_s, combined=ba_c, length_only=ba_len,
                increment_over_slop=ba_c - ba_s, increment_ci=[ilo, ihi],
                mean_words={a: float(d[d["class"] == a]["word_count"].mean()),
                            b: float(d[d["class"] == b]["word_count"].mean())})


def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    pg = build_pg(nlp)
    print(f"[pg] built {len(pg)} PG passages from {pg.source_id.nunique()} essays; "
          f"mean words={pg.word_count.mean():.0f}")
    existing = load_existing()
    dev = pd.concat([pg, existing], ignore_index=True)

    report = {"note": "EXPLORATORY; PG=nonfiction essay vs slop/modhuman=fiction (genre confound, M5)",
              "pg_passages": int(len(pg)), "pg_essays": int(pg.source_id.nunique()),
              "contrasts": {}}
    for a, b in [("pg", "slop"), ("pg", "modhuman"), ("pg", "great"), ("modhuman", "slop")]:
        report["contrasts"][f"{a}_vs_{b}"] = contrast(dev, a, b)

    with open(os.path.join(REPO, "results", "reports", "report_pg.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = ["\n## 2026-06-08 — EXPLORATORY: Paul Graham essays (high-quality modern human) vs AI/others",
             f"- PG: {len(pg)} passages / {pg.source_id.nunique()} essays, mean {pg.word_count.mean():.0f} words. "
             f"CONFOUND: PG=essay vs slop/modhuman=fiction (genre, M5). Not a binding test; held-out untouched."]
    for k, v in report["contrasts"].items():
        lines.append(
            f"- {k}: prosody_only={v['prosody_only']:.3f} CI[{v['prosody_ci'][0]:.3f},{v['prosody_ci'][1]:.3f}] "
            f"p={v['prosody_perm_p']:.3f} | slop_lex={v['slop_only']:.3f} | length_only={v['length_only']:.3f} | "
            f"incr_over_slop={v['increment_over_slop']:+.3f} CI{[round(x,3) for x in v['increment_ci']]} "
            f"(n={v['n']},g={v['n_groups']}, words {v['mean_words']})")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
