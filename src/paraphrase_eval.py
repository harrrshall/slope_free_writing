#!/usr/bin/env python3
"""Track A — paraphrase-robustness test (P6, pre-registered EXPERIMENT_LOG 2026-06-08).

Hypothesis: prosody is a paraphrase-robust ORIGIN signal. Paraphrasing AI text launders the lexical
slop n-grams (a lexical-slop detector degrades), but rhythmic structure is a more stable style property
(a prosody detector degrades LESS). Threat model: an adversary paraphrases to evade AI detection.

Design (paired, group-aware, no leakage): train each detector on ORIGINAL human-vs-AI features; in each
StratifiedGroupKFold fold, score the held-out groups' passages in BOTH original and PARAPHRASED form (the
test groups were never trained on). DROP = ba_original − ba_paraphrased. BINDING metric = (lexical_drop −
prosody_drop) with a paired group-bootstrap CI. P6 SUPPORTED iff lexical drops materially and the
difference-in-drops CI lower bound > 0 (prosody more robust); REFUTED iff prosody drops >= lexical.

Deterministic. Reads the paraphrased corpus the workflow wrote to data/passages_para/<class>/<id>.txt.
"""
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
CLASSES = ["modhuman", "slop"]
PARA_DIR = os.path.join(REPO, "data", "passages_para")


def _lr():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])


def extract_paraphrased(nlp):
    """Extract prosody (PROSODY_COLS) + lexical (SLOP_BASELINE_COLS) features on the paraphrased passages,
    the SAME way the originals were built. Returns (df, coverage_report)."""
    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    m = man[(man.band == "main") & (man["class"].isin(CLASSES))]
    rows, missing, lengths = [], [], []
    for _, r in m.iterrows():
        pid, cls, src = r["passage_id"], r["class"], r["source_id"]
        path = os.path.join(PARA_DIR, cls, pid + ".txt")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        with open(path) as f:
            text = f.read().strip()
        if len(text.split()) < 20:           # degenerate paraphrase guard
            missing.append(pid + "(too_short)")
            continue
        feats = extract_features(text, nlp, tier="A")     # prosody + mattr
        feats.update(lexical_features(text, nlp))         # lexical incl SLOP_BASELINE_COLS
        feats.update(passage_id=pid, **{"class": cls}, source_id=src, para_words=len(text.split()))
        rows.append(feats)
        lengths.append(len(text.split()))
    df = pd.DataFrame(rows)
    cov = {"n_para": len(df), "n_missing": len(missing), "missing": missing[:20]}
    return df, cov


def paired_oof(cols, orig, para, y, groups):
    """Train on ORIGINAL train fold; predict the test fold in BOTH original and paraphrased form.
    Returns (oof_orig, oof_para) aligned to the row order of `orig`."""
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof_o = np.empty(len(y), dtype=object)
    oof_p = np.empty(len(y), dtype=object)
    Xo = orig[cols].to_numpy(float)
    Xp = para[cols].to_numpy(float)
    for tr, te in skf.split(Xo, y, groups):
        model = _lr().fit(Xo[tr], y[tr])
        oof_o[te] = model.predict(Xo[te])
        oof_p[te] = model.predict(Xp[te])
    return oof_o.astype(str), oof_p.astype(str)


def group_bootstrap_diff_in_drops(y, oofs, groups, n=2000, seed=0):
    """Paired group bootstrap of (lexical_drop − prosody_drop). oofs = dict of 4 OOF arrays."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    uniq = pd.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}

    def ba(pred, idx):
        return balanced_accuracy_score(y[idx], np.asarray(pred)[idx])

    diffs, lex_drops, pros_drops = [], [], []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in samp])
        if len(set(y[idx])) < 2:
            continue
        ld = ba(oofs["lex_orig"], idx) - ba(oofs["lex_para"], idx)
        pd_ = ba(oofs["pros_orig"], idx) - ba(oofs["pros_para"], idx)
        lex_drops.append(ld); pros_drops.append(pd_); diffs.append(ld - pd_)
    pc = lambda a, q: float(np.percentile(a, q))
    return dict(diff_mean=float(np.mean(diffs)), diff_ci=[pc(diffs, 2.5), pc(diffs, 97.5)],
                lex_drop_ci=[pc(lex_drops, 2.5), pc(lex_drops, 97.5)],
                pros_drop_ci=[pc(pros_drops, 2.5), pc(pros_drops, 97.5)])


def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    para, cov = extract_paraphrased(nlp)
    print(f"[para] coverage: {cov['n_para']} paraphrased, {cov['n_missing']} missing")

    # ORIGINAL features for the same passages (prosody from features_A_p1c, lexical from lexical_A_p1c)
    of = pd.read_parquet(os.path.join(REPO, "results", "features", "features_A_p1c.parquet"))
    ol = pd.read_parquet(os.path.join(REPO, "results", "features", "lexical_A_p1c.parquet"))
    orig = of.merge(ol[["passage_id"] + [c for c in SLOP_BASELINE_COLS if c in ol.columns]], on="passage_id")
    orig = orig[(orig.band == "main") & (orig["class"].isin(CLASSES))]

    # align originals and paraphrases on the passages we actually paraphrased
    common = sorted(set(orig.passage_id) & set(para.passage_id))
    orig = orig[orig.passage_id.isin(common)].set_index("passage_id").loc[common].reset_index()
    para = para[para.passage_id.isin(common)].set_index("passage_id").loc[common].reset_index()
    y = orig["class"].to_numpy()
    groups = orig["source_id"].to_numpy()

    # length-preservation validity gate
    orig_words = orig["word_count"].to_numpy(float) if "word_count" in orig else None
    len_ratio = float(np.mean(para["para_words"].to_numpy(float) /
                              (orig_words if orig_words is not None else para["para_words"].to_numpy(float))))

    pros = [c for c in PROSODY_COLS if c in orig.columns]
    lex = [c for c in SLOP_BASELINE_COLS if c in orig.columns]
    oof_pros_o, oof_pros_p = paired_oof(pros, orig, para, y, groups)
    oof_lex_o, oof_lex_p = paired_oof(lex, orig, para, y, groups)

    def ba(p):
        return float(balanced_accuracy_score(y, p))
    res = {
        "n": len(common), "n_groups": int(pd.unique(groups).size), "coverage": cov,
        "mean_para_to_orig_length_ratio": len_ratio,
        "prosody": {"ba_original": ba(oof_pros_o), "ba_paraphrased": ba(oof_pros_p),
                    "drop": ba(oof_pros_o) - ba(oof_pros_p)},
        "lexical": {"ba_original": ba(oof_lex_o), "ba_paraphrased": ba(oof_lex_p),
                    "drop": ba(oof_lex_o) - ba(oof_lex_p)},
    }
    boot = group_bootstrap_diff_in_drops(
        y, {"pros_orig": oof_pros_o, "pros_para": oof_pros_p, "lex_orig": oof_lex_o, "lex_para": oof_lex_p},
        groups)
    res["difference_in_drops_lexical_minus_prosody"] = {"mean": boot["diff_mean"], "ci": boot["diff_ci"]}
    res["lex_drop_ci"], res["pros_drop_ci"] = boot["lex_drop_ci"], boot["pros_drop_ci"]

    # pre-registered validity gates + verdict
    lex_drop = res["lexical"]["drop"]
    validity_ok = (lex_drop > 0.03) and (0.6 <= len_ratio <= 1.6) and (cov["n_para"] >= 140)
    supported = validity_ok and (boot["diff_ci"][0] > 0)
    res["validity_gates"] = {"lexical_actually_degraded": bool(lex_drop > 0.03),
                             "length_preserved": bool(0.6 <= len_ratio <= 1.6),
                             "coverage_ok": bool(cov["n_para"] >= 140), "all_ok": bool(validity_ok)}
    res["verdict"] = "P6-SUPPORTED" if supported else ("INVALID" if not validity_ok else "P6-NOT-SUPPORTED")

    with open(os.path.join(REPO, "results", "reports", "report_paraphrase.json"), "w") as f:
        json.dump(res, f, indent=2)

    lines = [
        f"\n## 2026-06-08 — Track A RESULT (paraphrase-robustness, P6): VERDICT {res['verdict']}",
        f"- modhuman-vs-slop, n={res['n']} groups={res['n_groups']}, para coverage {cov['n_para']}/160, length ratio {len_ratio:.2f}",
        f"- PROSODY: ba original={res['prosody']['ba_original']:.3f} -> paraphrased={res['prosody']['ba_paraphrased']:.3f} (drop {res['prosody']['drop']:+.3f})",
        f"- LEXICAL: ba original={res['lexical']['ba_original']:.3f} -> paraphrased={res['lexical']['ba_paraphrased']:.3f} (drop {res['lexical']['drop']:+.3f})",
        f"- DIFFERENCE IN DROPS (lexical − prosody) = {boot['diff_mean']:+.3f} CI[{boot['diff_ci'][0]:.3f},{boot['diff_ci'][1]:.3f}] -> {'prosody MORE robust' if supported else 'NOT distinguishable / prosody not more robust'}",
        f"- validity gates: {res['validity_gates']}",
    ]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
