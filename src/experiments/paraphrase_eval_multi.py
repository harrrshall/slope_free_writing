#!/usr/bin/env python3
"""Track A POWER-UP: multi-realization paraphrase-robustness (P6), pre-registered EXPERIMENT_LOG 2026-06-08.

Fixes the M15 mistake (the original run used ONE paraphrase realization). Averages 3 independent paraphrase
realizations of the SAME 160 passages, folding the across-realization variance into the GROUP bootstrap (M11
kept — no switch to passage-level). Binding = difference-in-drops (lexical - prosody) CI lower bound > 0.

Reuses the validated paired_oof from paraphrase_eval.py (train detector on ORIGINAL, score paraphrased).
"""
import json
import os
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import balanced_accuracy_score

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ root on path for library imports
from features_lib import extract_features, PROSODY_COLS
from lexical_features import lexical_features, SLOP_BASELINE_COLS
from paraphrase_eval import paired_oof

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSES = ["modhuman", "slop"]
REAL_DIRS = [os.path.join(REPO, "data", "passages_para"),        # realization 1 (original run)
             os.path.join(REPO, "data", "passages_para_r2"),
             os.path.join(REPO, "data", "passages_para_r3")]


def extract(dirpath, nlp):
    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    m = man[(man.band == "main") & (man["class"].isin(CLASSES))]
    rows, missing = [], 0
    for _, r in m.iterrows():
        path = os.path.join(dirpath, r["class"], r["passage_id"] + ".txt")
        if not os.path.exists(path):
            missing += 1
            continue
        text = open(path).read().strip()
        if len(text.split()) < 20:
            missing += 1
            continue
        feats = extract_features(text, nlp, tier="A")
        feats.update(lexical_features(text, nlp))
        feats.update(passage_id=r["passage_id"], **{"class": r["class"]},
                     source_id=r["source_id"], para_words=len(text.split()))
        rows.append(feats)
    return pd.DataFrame(rows), missing


def group_bootstrap_multi(y, groups, pros_oo, pros_para, lex_oo, lex_para, n=2000, seed=0):
    """Resample GROUPS; per resample, average the realizations' paraphrased ba, compute drops, diff."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    uniq = pd.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}

    def ba(p, idx):
        return balanced_accuracy_score(y[idx], np.asarray(p)[idx])

    diffs, lexd, prosd = [], [], []
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(set(y[idx])) < 2:
            continue
        dp = ba(pros_oo, idx) - np.mean([ba(op, idx) for op in pros_para])
        dl = ba(lex_oo, idx) - np.mean([ba(op, idx) for op in lex_para])
        diffs.append(dl - dp); lexd.append(dl); prosd.append(dp)
    pc = lambda a, q: float(np.percentile(a, q))
    return dict(diff_mean=float(np.mean(diffs)), diff_ci=[pc(diffs, 2.5), pc(diffs, 97.5)],
                lex_drop_ci=[pc(lexd, 2.5), pc(lexd, 97.5)], pros_drop_ci=[pc(prosd, 2.5), pc(prosd, 97.5)])


def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    of = pd.read_parquet(os.path.join(REPO, "results", "features", "features_A_p1c.parquet"))
    ol = pd.read_parquet(os.path.join(REPO, "results", "features", "lexical_A_p1c.parquet"))
    orig = of.merge(ol[["passage_id"] + [c for c in SLOP_BASELINE_COLS if c in ol.columns]], on="passage_id")
    orig = orig[(orig.band == "main") & (orig["class"].isin(CLASSES))]

    paras, cov = [], []
    for d in REAL_DIRS:
        if not os.path.isdir(d):
            continue
        p, miss = extract(d, nlp)
        if len(p):
            paras.append(p); cov.append({"dir": os.path.basename(d), "n": int(len(p)), "missing": int(miss)})
    if len(paras) < 2:
        raise SystemExit(f"[multi] need >=2 realizations, found {len(paras)} ({cov})")

    common = set(orig.passage_id)
    for p in paras:
        common &= set(p.passage_id)
    common = sorted(common)
    orig = orig.set_index("passage_id").loc[common].reset_index()
    paras = [p.set_index("passage_id").loc[common].reset_index() for p in paras]
    y, g = orig["class"].to_numpy(), orig["source_id"].to_numpy()
    pros = [c for c in PROSODY_COLS if c in orig.columns]
    lex = [c for c in SLOP_BASELINE_COLS if c in orig.columns]

    pros_oo = lex_oo = None
    pros_para, lex_para = [], []
    for p in paras:
        oo, op = paired_oof(pros, orig, p, y, g); pros_oo = oo; pros_para.append(op)
        oo2, op2 = paired_oof(lex, orig, p, y, g); lex_oo = oo2; lex_para.append(op2)

    def ba(pred):
        return float(balanced_accuracy_score(y, pred))
    ba_pros_orig, ba_lex_orig = ba(pros_oo), ba(lex_oo)
    ba_pros_para_r = [ba(op) for op in pros_para]
    ba_lex_para_r = [ba(op) for op in lex_para]
    drop_pros = ba_pros_orig - float(np.mean(ba_pros_para_r))
    drop_lex = ba_lex_orig - float(np.mean(ba_lex_para_r))
    boot = group_bootstrap_multi(y, g, pros_oo, pros_para, lex_oo, lex_para)

    # validity (per realization)
    ratios = [float((p["para_words"].to_numpy(float) /
                     orig["word_count"].to_numpy(float)).mean()) for p in paras]
    lexical_degraded = drop_lex > 0.03
    length_ok = all(0.6 <= r <= 1.6 for r in ratios)
    coverage_ok = all(c["n"] >= 140 for c in cov)
    valid = lexical_degraded and length_ok and coverage_ok
    supported = valid and boot["diff_ci"][0] > 0

    res = {"n": len(common), "n_groups": int(pd.unique(g).size), "n_realizations": len(paras), "coverage": cov,
           "length_ratios": [round(r, 3) for r in ratios],
           "prosody": {"ba_original": ba_pros_orig, "ba_paraphrased_per_realization": [round(x, 3) for x in ba_pros_para_r],
                       "ba_paraphrased_mean": float(np.mean(ba_pros_para_r)), "drop": drop_pros},
           "lexical": {"ba_original": ba_lex_orig, "ba_paraphrased_per_realization": [round(x, 3) for x in ba_lex_para_r],
                       "ba_paraphrased_mean": float(np.mean(ba_lex_para_r)), "drop": drop_lex},
           "difference_in_drops_lexical_minus_prosody": {"mean": boot["diff_mean"], "ci": boot["diff_ci"]},
           "lex_drop_ci": boot["lex_drop_ci"], "pros_drop_ci": boot["pros_drop_ci"],
           "validity_gates": {"lexical_actually_degraded": bool(lexical_degraded), "length_preserved": bool(length_ok),
                              "coverage_ok": bool(coverage_ok), "all_ok": bool(valid)},
           "verdict": "P6-SUPPORTED" if supported else ("INVALID" if not valid else "P6-NOT-SUPPORTED"),
           "ai_ceiling_note": "slop side limited to 15 generator-model groups (lars1234 has exactly 15); CI is group-limited on the AI side regardless of passage count"}
    with open(os.path.join(REPO, "results", "reports", "report_paraphrase_multi.json"), "w") as f:
        json.dump(res, f, indent=2)

    lines = [f"\n## 2026-06-08 — Track A POWER-UP RESULT (multi-realization, {len(paras)} realizations): VERDICT {res['verdict']}",
             f"- modhuman-vs-slop, n={res['n']} groups={res['n_groups']}, realizations={len(paras)}, length ratios={res['length_ratios']}",
             f"- PROSODY: ba_orig={ba_pros_orig:.3f} -> para_mean={np.mean(ba_pros_para_r):.3f} (per-real {[round(x,3) for x in ba_pros_para_r]}); drop {drop_pros:+.3f}",
             f"- LEXICAL: ba_orig={ba_lex_orig:.3f} -> para_mean={np.mean(ba_lex_para_r):.3f} (per-real {[round(x,3) for x in ba_lex_para_r]}); drop {drop_lex:+.3f}",
             f"- DIFFERENCE IN DROPS (lexical-prosody) = {boot['diff_mean']:+.3f} CI[{boot['diff_ci'][0]:.3f},{boot['diff_ci'][1]:.3f}] -> {'prosody MORE robust (CI>0)' if supported else 'CI grazes/includes 0'}",
             f"- validity {res['validity_gates']} | AI ceiling: 15 generator models (group-limited, disclosed)"]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
