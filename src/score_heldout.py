#!/usr/bin/env python3
"""Score the quarantined held-out EXACTLY ONCE (only if the gate said PROCEED).

Trains on DEV, predicts HELD-OUT (binary great-vs-flat), reports balanced_accuracy +
bootstrap CI, grouped permutation_importance computed ON the held-out, and the depth-3
tree text. Writes heldout_report.json.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
from features_lib import PROSODY_COLS, TIER_B_COLS, FEATURE_GROUPS, RHYTHM_GROUPS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bootstrap_ci(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    s = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        s.append(balanced_accuracy_score(y[idx], pred[idx]))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def run_p1_heldout(features, lexical, split_path, out):
    """Phase-1 held-out, scored ONCE: the new genre/era-matched contrasts + lexical-residualized prosody."""
    from evaluate import group_bootstrap_ci, Residualizer
    from lexical_features import LEXICAL_COLS, SLOP_BASELINE_COLS
    df = pd.read_parquet(os.path.join(REPO, features))
    lex = pd.read_parquet(os.path.join(REPO, lexical))
    split = json.load(open(os.path.join(REPO, split_path)))
    pros = [c for c in PROSODY_COLS if c in df.columns]
    lexcols = [c for c in LEXICAL_COLS if c in lex.columns]
    dfp = df.drop(columns=[c for c in ["mattr"] if c in df.columns])
    merged = dfp.merge(lex[["passage_id"] + lexcols], on="passage_id", how="inner")
    dev = merged[(merged.passage_id.isin(set(split["dev"]))) & (merged.band == "main")]
    held = merged[(merged.passage_id.isin(set(split["held_out"]))) & (merged.band == "main")]

    def lr():
        return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])

    def pair(a, b):
        tr, te = dev[dev["class"].isin([a, b])], held[held["class"].isin([a, b])]
        if te["class"].nunique() < 2:
            return None
        p = lr().fit(tr[pros], tr["class"]); pred = p.predict(te[pros])
        lo, hi = group_bootstrap_ci(te["class"].to_numpy(), pred, te["source_id"].to_numpy())
        return dict(ba=float(balanced_accuracy_score(te["class"], pred)), ci_low=lo, ci_high=hi, n=int(len(te)))

    res = {"modhuman_vs_slop": pair("modhuman", "slop"),
           "great_vs_modhuman": pair("great", "modhuman"),
           "great_vs_slop": pair("great", "slop")}

    slop = [c for c in SLOP_BASELINE_COLS if c in merged.columns]

    def resid_pair(a, b):
        """SLOP-baseline-residualized prosody, trained on dev, scored ONCE on held-out (H2b.2 confirm)."""
        tr, te = dev[dev["class"].isin([a, b])], held[held["class"].isin([a, b])]
        if te["class"].nunique() < 2:
            return None
        cols = pros + slop
        pipe = Pipeline([("resid", Residualizer(list(range(len(pros), len(cols))), list(range(len(pros))))),
                         ("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(tr[cols], tr["class"])
        pred = pipe.predict(te[cols])
        lo, hi = group_bootstrap_ci(te["class"].to_numpy(), pred, te["source_id"].to_numpy())
        return dict(ba=float(balanced_accuracy_score(te["class"], pred)), ci_low=lo, ci_high=hi, n=int(len(te)))

    res["great_vs_modhuman_slopresid"] = resid_pair("great", "modhuman")   # BINDING H2b.2
    res["great_vs_slop_slopresid"] = resid_pair("great", "slop")
    json.dump(res, open(os.path.join(REPO, out), "w"), indent=2)
    print("[heldout-p1]", json.dumps({k: (round(v["ba"], 3) if v else None) for k, v in res.items()}))
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write("\n- P1 HELD-OUT (once): " + ", ".join(
            f"{k} ba={v['ba']:.3f} CI[{v['ci_low']:.3f},{v['ci_high']:.3f}]" for k, v in res.items() if v) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexical", default=None)
    args = ap.parse_args()
    if args.lexical:
        return run_p1_heldout(args.features, args.lexical, args.split, args.out)

    df = pd.read_parquet(os.path.join(REPO, args.features))
    split = json.load(open(os.path.join(REPO, args.split)))
    dev_ids, held_ids = set(split["dev"]), set(split["held_out"])
    pros = [c for c in PROSODY_COLS + TIER_B_COLS if c in df.columns]

    def binary(ids, classes):
        d = df[(df.passage_id.isin(ids)) & (df.band == "main") & (df["class"].isin(classes))]
        return d[pros].to_numpy(float), d["class"].to_numpy()

    # held-out is touched in this single script run only (scored once).
    pairwise = {}
    for classes in (["great", "flat"], ["great", "slop"], ["flat", "slop"]):
        Xtr, ytr = binary(dev_ids, classes)
        Xte, yte = binary(held_ids, classes)
        p = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(Xtr, ytr)
        pr = p.predict(Xte)
        blo, bhi = bootstrap_ci(yte, pr)
        pairwise["_vs_".join(classes)] = dict(ba=float(balanced_accuracy_score(yte, pr)),
                                              ci_low=blo, ci_high=bhi, n=int(len(yte)))

    Xtr, ytr = binary(dev_ids, ["great", "flat"])
    Xte, yte = binary(held_ids, ["great", "flat"])
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    ba = balanced_accuracy_score(yte, pred)
    lo, hi = bootstrap_ci(yte, pred)

    pi = permutation_importance(pipe, Xte, yte, n_repeats=50, random_state=0, scoring="balanced_accuracy")
    feat_to_group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    grp = {}
    for col, imp in zip(pros, pi.importances_mean):
        g = feat_to_group.get(col)
        if g:
            grp[g] = grp.get(g, 0.0) + float(imp)
    ranked = sorted(grp.items(), key=lambda x: -x[1])

    tr = Pipeline([("sc", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))]).fit(Xtr, ytr)
    tree_txt = export_text(tr.named_steps["clf"], feature_names=list(pros))

    out = {"heldout_binary_ba": float(ba), "ci_low": lo, "ci_high": hi,
           "n_heldout": int(len(yte)), "pairwise_heldout": pairwise,
           "importance_groups": {g: round(v, 5) for g, v in ranked},
           "top3_groups": [g for g, _ in ranked[:3]],
           "rhythm_group_in_top3": any(g in RHYTHM_GROUPS for g, _ in ranked[:3]),
           "tree_text": tree_txt}
    with open(os.path.join(REPO, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[heldout] great-vs-flat ba={ba:.3f} CI[{lo:.3f},{hi:.3f}] n={len(yte)}")
    print(f"[heldout] pairwise: {json.dumps({k: round(v['ba'],3) for k,v in pairwise.items()})}")
    print(f"[heldout] importance groups: {ranked}")
    print(tree_txt)


if __name__ == "__main__":
    main()
