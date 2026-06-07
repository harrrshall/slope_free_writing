#!/usr/bin/env python3
"""Phase 0 evaluation on the DEV set only (held-out is quarantined).

Every confound control is FOLD-INTERNAL (refit per CV fold) so a positive result
cannot be a leak. Produces report_<tier>.json + appends raw numbers to EXPERIMENT_LOG.md.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (StratifiedGroupKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

from features_lib import PROSODY_COLS, TIER_B_COLS, FEATURE_GROUPS, RHYTHM_GROUPS
from lexical_features import LEXICAL_COLS, DISTINCT_COLS, DIVERSITY_COLS, SLOP_BASELINE_COLS

REPO = os.path.dirname(os.path.abspath(__file__))


def logreg():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000))])


def tree():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))])


def cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


class Residualizer(BaseEstimator, TransformerMixin):
    """Regress covariate columns out of feature columns; FIT ON TRAIN FOLD ONLY."""
    def __init__(self, cov_idx, feat_idx):
        self.cov_idx = cov_idx
        self.feat_idx = feat_idx

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.models_ = []
        C = X[:, self.cov_idx]
        for j in self.feat_idx:
            m = LinearRegression().fit(C, X[:, j])
            self.models_.append(m)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        C = X[:, self.cov_idx]
        return np.column_stack([X[:, j] - m.predict(C)
                                for j, m in zip(self.feat_idx, self.models_)])


def group_perm_test(model, X, y, groups, n=1000, seed=0):
    """GROUP-LEVEL permutation test (the correct null for constant-label groups).

    Each group (book/category/model) has a single class label, so sklearn's default
    permutation_test_score (which shuffles labels WITHIN groups) is degenerate -> p=1.0.
    Here we permute the per-GROUP label assignment, keeping each group's samples together,
    which is the honest null: 'could features separate classes this well if whole groups
    were randomly labelled?'
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); groups = np.asarray(groups)
    uniq = pd.unique(groups)
    g2y = {gg: y[groups == gg][0] for gg in uniq}
    base = np.array([g2y[gg] for gg in uniq])
    true = float(cross_val_score(clone(model), X, y, groups=groups, cv=cv(),
                                 scoring="balanced_accuracy").mean())
    perms, ge = [], 0
    for _ in range(n):
        m = dict(zip(uniq, rng.permutation(base)))
        yp = np.array([m[gg] for gg in groups])
        try:
            sc = float(cross_val_score(clone(model), X, yp, groups=groups, cv=cv(),
                                       scoring="balanced_accuracy").mean())
        except ValueError:
            continue
        perms.append(sc); ge += (sc >= true - 1e-12)
    valid = len(perms)
    p = (ge + 1) / (valid + 1)
    return dict(perm_true=true, perm_p=float(p), perm_mean=float(np.mean(perms)),
                perm_max=float(np.max(perms)), perm_n=valid)


def bootstrap_ci(y, oof, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); oof = np.asarray(oof)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        scores.append(balanced_accuracy_score(y[idx], oof[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def eval_clf(model, X, y, groups, do_perm=False, perm_n=1000):
    oof = cross_val_predict(clone(model), X, y, groups=groups, cv=cv())
    ba = balanced_accuracy_score(y, oof)
    lo, hi = bootstrap_ci(y, oof)
    f1 = f1_score(y, oof, average="macro")
    labels = sorted(set(y))
    cm = confusion_matrix(y, oof, labels=labels).tolist()
    out = dict(balanced_acc=float(ba), ci_low=lo, ci_high=hi, macro_f1=float(f1),
               labels=labels, confusion=cm)
    if do_perm:
        pt = group_perm_test(model, X, y, groups, n=perm_n)
        out.update(perm_score=pt["perm_true"], perm_p=pt["perm_p"],
                   perm_mean=pt["perm_mean"], perm_max=pt["perm_max"], perm_n=pt["perm_n"])
    return out


# ====================== Phase 1 (H2 orthogonality + H3 genre/era) ======================
def group_bootstrap_ci(y, oof, groups, n=2000, seed=0):
    """Bootstrap CI resampling WHOLE source_id groups (labels are group-constant -> passage-level
    bootstrap underestimates uncertainty; the gate requires group-level CIs)."""
    rng = np.random.default_rng(seed)
    y, oof, groups = np.asarray(y), np.asarray(oof), np.asarray(groups)
    uniq = pd.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    s = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in samp])
        if len(set(y[idx])) < 2:
            continue
        s.append(balanced_accuracy_score(y[idx], oof[idx]))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def group_bootstrap_increment_ci(y, oof_c, oof_l, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, oc, ol, groups = np.asarray(y), np.asarray(oof_c), np.asarray(oof_l), np.asarray(groups)
    uniq = pd.unique(groups); idx_by = {g: np.where(groups == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in samp])
        if len(set(y[idx])) < 2:
            continue
        d.append(balanced_accuracy_score(y[idx], oc[idx]) - balanced_accuracy_score(y[idx], ol[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d))


def _oof_ba(model, X, y, g):
    oof = cross_val_predict(clone(model), X, y, groups=g, cv=cv())
    return oof, float(balanced_accuracy_score(y, oof))


def lexical_residualized_prosody(d, pros, lexcols, perm_n=1000):
    """Regress lexical OUT of prosody fold-internally, then classify on the residuals (H2.1)."""
    cols = pros + lexcols
    X = d[cols].to_numpy(float); y = d["class"].to_numpy(); g = d["source_id"].to_numpy()
    cov_idx = list(range(len(pros), len(pros) + len(lexcols)))
    feat_idx = list(range(len(pros)))
    pipe = Pipeline([("resid", Residualizer(cov_idx, feat_idx)),
                     ("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
    oof = cross_val_predict(pipe, X, y, groups=g, cv=cv())
    ba = balanced_accuracy_score(y, oof)
    lo, hi = group_bootstrap_ci(y, oof, g)
    pt = group_perm_test(pipe, X, y, g, n=perm_n)
    return dict(balanced_acc=float(ba), ci_low=lo, ci_high=hi, perm_p=pt["perm_p"], perm_mean=pt["perm_mean"])


def h2_contrast(dev, pros, lexcols, a, b, perm_n=1000):
    """Corrected (Phase 1b): PRIMARY baseline = SLOP_BASELINE_COLS (genuine slop lexicon + readability).
    Full lexical (incl. diversity proxies) reported as a transparency diagnostic only."""
    d = dev[dev["class"].isin([a, b])].reset_index(drop=True)
    if d["class"].nunique() < 2:
        return None
    y = d["class"].to_numpy(); g = d["source_id"].to_numpy()
    slop = [c for c in SLOP_BASELINE_COLS if c in d.columns]
    _, ba_p = _oof_ba(logreg(), d[pros].to_numpy(float), y, g)
    oof_ls, ba_ls = _oof_ba(logreg(), d[slop].to_numpy(float), y, g)          # slop-only baseline
    oof_lf, ba_lf = _oof_ba(logreg(), d[lexcols].to_numpy(float), y, g)       # full lexical (diagnostic)
    oof_c, ba_c = _oof_ba(logreg(), d[pros + slop].to_numpy(float), y, g)     # prosody + slop baseline
    ilo, ihi, imean = group_bootstrap_increment_ci(y, oof_c, oof_ls, g)
    return dict(contrast=f"{a}_vs_{b}", n=int(len(d)), n_groups=int(pd.unique(g).size),
                prosody_only=ba_p, lexical_slop_only=ba_ls, lexical_full_only=ba_lf,
                combined_slop=ba_c, increment_over_slop=float(ba_c - ba_ls),
                increment_ci=[ilo, ihi], increment_mean=imean,
                resid_slop=lexical_residualized_prosody(d, pros, slop, perm_n),        # PRIMARY (binding)
                resid_full=lexical_residualized_prosody(d, pros, lexcols, perm_n))     # diagnostic


def h3_contrast(dev, cols, a, b, src_exclude=None, perm_n=1000, do_perm=True):
    d = dev[dev["class"].isin([a, b])].copy()
    if src_exclude:
        d = d[~d["source_id"].astype(str).str.contains(src_exclude, case=False, regex=True)]
    d = d.reset_index(drop=True)
    if d["class"].nunique() < 2:
        return None
    y = d["class"].to_numpy(); g = d["source_id"].to_numpy()
    oof, ba = _oof_ba(logreg(), d[cols].to_numpy(float), y, g)
    lo, hi = group_bootstrap_ci(y, oof, g)
    out = dict(contrast=f"{a}_vs_{b}", n=int(len(d)), n_groups=int(pd.unique(g).size),
               balanced_acc=ba, ci_low=lo, ci_high=hi)
    if do_perm:
        out["perm_p"] = group_perm_test(logreg(), d[cols].to_numpy(float), y, g, n=perm_n)["perm_p"]
    return out


def run_phase1(df, split, args):
    lex = pd.read_parquet(os.path.join(REPO, args.lexical))
    pros = [c for c in PROSODY_COLS if c in df.columns]
    lexcols = [c for c in LEXICAL_COLS if c in lex.columns]
    subsent = [c for c in pros if c not in ("sl_mean", "sl_var", "sl_cv")]
    dfp = df.drop(columns=[c for c in ["mattr"] if c in df.columns])    # mattr is lexical's (D11)
    merged = dfp.merge(lex[["passage_id"] + lexcols], on="passage_id", how="inner")
    dev = merged[(merged.passage_id.isin(set(split["dev"]))) & (merged.band == "main")].reset_index(drop=True)
    present = set(dev["class"])
    report = {"pros": pros, "lexcols": lexcols, "classes": sorted(present),
              "dev_counts": dev["class"].value_counts().to_dict(), "h2": {}, "h3": {}}

    report["h2"]["great_vs_slop"] = h2_contrast(dev, pros, lexcols, "great", "slop")
    if "modhuman" in present:
        report["h2"]["modhuman_vs_slop"] = h2_contrast(dev, pros, lexcols, "modhuman", "slop")
        report["h2"]["great_vs_modhuman"] = h2_contrast(dev, pros, lexcols, "great", "modhuman")  # BINDING (H2b.1)

    if not args.h2_only:
        if "modhuman" in present:
            report["h3"]["modhuman_vs_slop"] = h3_contrast(dev, pros, "modhuman", "slop")          # BINDING
            report["h3"]["great_vs_modhuman"] = h3_contrast(dev, pros, "great", "modhuman")        # era within narrative
            report["h3"]["great_vs_modhuman_LEXICALonly"] = h3_contrast(dev, lexcols, "great", "modhuman", do_perm=False)
        if "modgreat" in present:
            report["h3"]["great_vs_modgreat_full"] = h3_contrast(dev, pros, "great", "modgreat", do_perm=False)
            report["h3"]["modgreat_vs_slop_subsent_noHF"] = h3_contrast(dev, subsent, "modgreat", "slop", src_exclude="Hemingway|Faulkner", do_perm=False)
            report["h3"]["modgreat_vs_great_subsent_noHF"] = h3_contrast(dev, subsent, "modgreat", "great", src_exclude="Hemingway|Faulkner", do_perm=False)

    with open(os.path.join(REPO, args.report), "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"\n## EVAL-P1 {args.features} + {args.lexical} -> {args.report} ({'H2-only' if args.h2_only else 'H2+H3'})"]
    for ck, v in report["h2"].items():
        if not v:
            continue
        rs, rf = v["resid_slop"], v["resid_full"]
        binding = " [BINDING H2b.1]" if ck == "great_vs_modhuman" else ""
        lines.append(
            f"- H2 {ck}{binding}: prosody_only={v['prosody_only']:.3f} slop_baseline={v['lexical_slop_only']:.3f} "
            f"full_lexical={v['lexical_full_only']:.3f} increment_over_slop={v['increment_over_slop']:+.3f} CI{[round(x,3) for x in v['increment_ci']]}; "
            f"resid_vs_SLOP ba={rs['balanced_acc']:.3f} CI[{rs['ci_low']:.3f},{rs['ci_high']:.3f}] p={rs['perm_p']:.4f} | "
            f"resid_vs_FULL(diag) ba={rf['balanced_acc']:.3f} CI_low={rf['ci_low']:.3f}")
    for k, v in report["h3"].items():
        if v:
            extra = f" perm_p={v['perm_p']:.4f}" if "perm_p" in v else ""
            lines.append(f"- H3 {k}: ba={v['balanced_acc']:.3f} CI[{v['ci_low']:.3f},{v['ci_high']:.3f}]{extra} (n={v['n']},g={v['n_groups']})")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--lexical", default=None)
    ap.add_argument("--h2-only", action="store_true", dest="h2_only")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(REPO, args.features))
    split = json.load(open(os.path.join(REPO, args.split)))
    if args.lexical:
        return run_phase1(df, split, args)
    dev_ids = set(split["dev"])

    pros = [c for c in PROSODY_COLS + TIER_B_COLS if c in df.columns]
    dev = df[(df.passage_id.isin(dev_ids)) & (df.band == "main")].reset_index(drop=True)

    report = {"tier_cols": pros, "n_dev": len(dev),
              "dev_class_counts": dev["class"].value_counts().to_dict()}

    # ---- binary great-vs-flat (the binding headline) ----
    b = dev[dev["class"].isin(["great", "flat"])].reset_index(drop=True)
    Xb, yb, gb = b[pros].to_numpy(float), b["class"].to_numpy(), b["source_id"].to_numpy()
    report["binary"] = {
        "logreg": eval_clf(logreg(), Xb, yb, gb, do_perm=True),
        "tree": eval_clf(tree(), Xb, yb, gb, do_perm=False),
    }

    # ---- 3-class ----
    X3, y3, g3 = dev[pros].to_numpy(float), dev["class"].to_numpy(), dev["source_id"].to_numpy()
    report["three_class"] = {
        "logreg": eval_clf(logreg(), X3, y3, g3, do_perm=True),
        "tree": eval_clf(tree(), X3, y3, g3, do_perm=False),
    }

    # ---- baselines (binary): length-only and MATTR-only ----
    def single_feat_ba(col):
        oof = cross_val_predict(logreg(), b[[col]].to_numpy(float), yb, groups=gb, cv=cv())
        return float(balanced_accuracy_score(yb, oof))
    report["baselines"] = {
        "length_only_ba": single_feat_ba("sl_mean"),
        "mattr_only_ba": single_feat_ba("mattr"),
    }
    report["prosody_minus_best_baseline"] = (
        report["binary"]["logreg"]["balanced_acc"]
        - max(report["baselines"]["length_only_ba"], report["baselines"]["mattr_only_ba"]))

    # ---- fold-internal residualized prosody (out: sl_mean + mattr) ----
    resid_cols = [c for c in pros if c != "sl_mean"]
    Xr = b[resid_cols + ["sl_mean", "mattr"]].to_numpy(float)
    cov_idx = [len(resid_cols), len(resid_cols) + 1]
    feat_idx = list(range(len(resid_cols)))
    resid_pipe = Pipeline([("resid", Residualizer(cov_idx, feat_idx)),
                           ("sc", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=3000))])
    oof_r = cross_val_predict(resid_pipe, Xr, yb, groups=gb, cv=cv())
    ba_r = balanced_accuracy_score(yb, oof_r)
    lo_r, hi_r = bootstrap_ci(yb, oof_r)
    report["residualized"] = {"balanced_acc": float(ba_r), "ci_low": lo_r, "ci_high": hi_r,
                              "covariates": ["sl_mean", "mattr"]}

    # ---- era-balanced re-run: great + euclaise(modern human) vs flat ----
    eb_great = pd.concat([dev[dev["class"] == "great"], df[df.band == "era_band"]], ignore_index=True)
    eb_great = eb_great.copy(); eb_great["class"] = "great"
    eb_great["source_id"] = [sid if sid != "euclaise" else f"euclaise_{i%5}"
                             for i, sid in enumerate(eb_great["source_id"])]
    eb = pd.concat([eb_great, dev[dev["class"] == "flat"]], ignore_index=True)
    Xe, ye, ge = eb[pros].to_numpy(float), eb["class"].to_numpy(), eb["source_id"].to_numpy()
    oof_e = cross_val_predict(logreg(), Xe, ye, groups=ge, cv=cv())
    report["era_balanced"] = {"binary_ba": float(balanced_accuracy_score(ye, oof_e)),
                              "n": len(eb), "n_euclaise": int((df.band == "era_band").sum())}

    # ---- grouped permutation importance (structural rhythm-in-top-3 check) ----
    feat_to_group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    fitted = logreg().fit(Xb, yb)
    pi = permutation_importance(fitted, Xb, yb, n_repeats=20, random_state=0,
                                scoring="balanced_accuracy")
    grp = {}
    for col, imp in zip(pros, pi.importances_mean):
        g = feat_to_group.get(col)
        if g:
            grp[g] = grp.get(g, 0.0) + float(imp)
    ranked = sorted(grp.items(), key=lambda x: -x[1])
    top3 = [g for g, _ in ranked[:3]]
    report["importance_groups"] = {g: round(v, 5) for g, v in ranked}
    report["top3_groups"] = top3
    report["rhythm_group_in_top3"] = any(g in RHYTHM_GROUPS for g in top3)

    with open(os.path.join(REPO, args.report), "w") as f:
        json.dump(report, f, indent=2)

    # ---- log raw numbers ----
    bl = report["binary"]["logreg"]
    summary = (
        f"\n## EVAL {args.features} -> {args.report}\n"
        f"- binary great-vs-flat (logreg): balanced_acc={bl['balanced_acc']:.3f} "
        f"CI[{bl['ci_low']:.3f},{bl['ci_high']:.3f}] perm_p={bl['perm_p']:.4f} macroF1={bl['macro_f1']:.3f}\n"
        f"- binary (tree): ba={report['binary']['tree']['balanced_acc']:.3f}\n"
        f"- 3-class (logreg): ba={report['three_class']['logreg']['balanced_acc']:.3f} "
        f"perm_p={report['three_class']['logreg']['perm_p']:.4f}\n"
        f"- baselines: length_only={report['baselines']['length_only_ba']:.3f} "
        f"mattr_only={report['baselines']['mattr_only_ba']:.3f}; "
        f"prosody - best_baseline={report['prosody_minus_best_baseline']:.3f}\n"
        f"- residualized(out sl_mean+mattr): ba={ba_r:.3f} CI[{lo_r:.3f},{hi_r:.3f}]\n"
        f"- era-balanced binary ba={report['era_balanced']['binary_ba']:.3f}\n"
        f"- importance groups (ranked): {ranked}\n"
        f"- rhythm group in top3: {report['rhythm_group_in_top3']} (top3={top3})\n")
    with open(os.path.join(REPO, "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
