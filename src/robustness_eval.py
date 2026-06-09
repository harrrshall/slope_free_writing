#!/usr/bin/env python3
"""General paraphrase-robustness eval (Track A generality, Z2). Tests whether a human-vs-AI detector's
accuracy DROP under a given paraphraser is SMALLER for PROSODY than for LEXICAL features.

Generalizes paraphrase_eval.py to ANY 2 classes, ANY orig/para dirs, ANY paraphraser, ANY manifest(s).
Extracts features FRESH (no parquet dependency) so it works on new piles (e.g. gpt4o). Group-bootstrap over
source_id (M11). Validity-gated. Each class arg = 'CLASS:ORIG_DIR:PARA_DIR' (text at <DIR>/<CLASS>/<id>.txt).

Sanity: --human modhuman:data/passages:data/passages_para --ai slop:data/passages:data/passages_para
        should reproduce F5 single-realization (diff-in-drops ~ +0.056).
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import spacy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

from features_lib import extract_features, PROSODY_COLS
from lexical_features import lexical_features, SLOP_BASELINE_COLS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lr():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])


def extract_dir(rows, dirpath, cls, nlp):
    out, missing = [], 0
    for _, r in rows.iterrows():
        path = os.path.join(REPO, dirpath, cls, r["passage_id"] + ".txt")
        if not os.path.exists(path):
            missing += 1
            continue
        t = open(path).read().strip()
        if len(t.split()) < 20:
            missing += 1
            continue
        f = extract_features(t, nlp, "A")
        f.update(lexical_features(t, nlp))
        f.update(passage_id=r["passage_id"], **{"class": cls}, source_id=str(r["source_id"]), words=len(t.split()))
        out.append(f)
    return pd.DataFrame(out), missing


def paired_oof(cols, orig, para, y, g):
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    oo, op = np.empty(len(y), object), np.empty(len(y), object)
    Xo, Xp = orig[cols].to_numpy(float), para[cols].to_numpy(float)
    for tr, te in skf.split(Xo, y, g):
        m = lr().fit(Xo[tr], y[tr])
        oo[te] = m.predict(Xo[te]); op[te] = m.predict(Xp[te])
    return oo.astype(str), op.astype(str)


def gboot(y, g, oofs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, g = np.asarray(y), np.asarray(g)
    uniq = pd.unique(g)
    idxby = {gg: np.where(g == gg)[0] for gg in uniq}

    def ba(p, idx):
        return balanced_accuracy_score(y[idx], np.asarray(p)[idx])
    diffs = []
    for _ in range(n):
        idx = np.concatenate([idxby[gg] for gg in rng.choice(uniq, len(uniq), replace=True)])
        if len(set(y[idx])) < 2:
            continue
        dl = ba(oofs["lex_o"], idx) - ba(oofs["lex_p"], idx)
        dp = ba(oofs["pros_o"], idx) - ba(oofs["pros_p"], idx)
        diffs.append(dl - dp)
    return float(np.mean(diffs)), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="CLASS:ORIG_DIR:PARA_DIR")
    ap.add_argument("--ai", required=True, help="CLASS:ORIG_DIR:PARA_DIR")
    ap.add_argument("--manifests", required=True, help="comma list of manifest csvs (for source_id)")
    ap.add_argument("--group-by", default="source_id", choices=["source_id", "passage_id"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    man = pd.concat([pd.read_csv(os.path.join(REPO, p)) for p in a.manifests.split(",")], ignore_index=True)

    of_all, pf_all = [], []
    for spec in (a.human, a.ai):
        cls, od, pdp = spec.split(":")
        rows = man[(man.band == "main") & (man["class"] == cls)][["passage_id", "source_id"]]
        of, _ = extract_dir(rows, od, cls, nlp)
        pf, _ = extract_dir(rows, pdp, cls, nlp)
        of_all.append(of); pf_all.append(pf)
    orig = pd.concat(of_all, ignore_index=True)
    para = pd.concat(pf_all, ignore_index=True)
    common = sorted(set(orig.passage_id) & set(para.passage_id))
    orig = orig.set_index("passage_id").loc[common].reset_index()
    para = para.set_index("passage_id").loc[common].reset_index()
    gcol = "passage_id" if a.group_by == "passage_id" else "source_id"
    y, g = orig["class"].to_numpy(), orig[gcol].to_numpy()
    pros = [c for c in PROSODY_COLS if c in orig.columns]
    lex = [c for c in SLOP_BASELINE_COLS if c in orig.columns]
    pros_o, pros_p = paired_oof(pros, orig, para, y, g)
    lex_o, lex_p = paired_oof(lex, orig, para, y, g)

    def ba(p):
        return float(balanced_accuracy_score(y, p))
    drop_pros, drop_lex = ba(pros_o) - ba(pros_p), ba(lex_o) - ba(lex_p)
    dm, dci = gboot(y, g, dict(pros_o=pros_o, pros_p=pros_p, lex_o=lex_o, lex_p=lex_p))
    oi, pi = orig.set_index("passage_id"), para.set_index("passage_id")
    lenratio = float((pi.loc[common, "words"].to_numpy(float) / oi.loc[common, "words"].to_numpy(float)).mean())
    valid = drop_lex > 0.03 and 0.6 <= lenratio <= 1.6 and len(common) >= 140
    supported = valid and dci[0] > 0
    res = dict(label=a.label, classes=[a.human.split(":")[0], a.ai.split(":")[0]], group_by=gcol,
               n=len(common), n_groups=int(pd.unique(g).size), length_ratio=round(lenratio, 3),
               prosody=dict(ba_orig=round(ba(pros_o), 3), ba_para=round(ba(pros_p), 3), drop=round(drop_pros, 3)),
               lexical=dict(ba_orig=round(ba(lex_o), 3), ba_para=round(ba(lex_p), 3), drop=round(drop_lex, 3)),
               diff_in_drops=dict(mean=round(dm, 3), ci=[round(dci[0], 3), round(dci[1], 3)]),
               validity=dict(lex_degraded=bool(drop_lex > 0.03), length_ok=bool(0.6 <= lenratio <= 1.6),
                             coverage_n=len(common), all_ok=bool(valid)),
               verdict="P6-SUPPORTED" if supported else ("INVALID" if not valid else "P6-NOT-SUPPORTED"))
    json.dump(res, open(os.path.join(REPO, a.out), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
