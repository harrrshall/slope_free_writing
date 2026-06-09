#!/usr/bin/env python3
"""Stage-1 driver: do ORDER-aware features add within-human quality signal beyond the order-blind
histograms AND the lexical-slop baseline? Binding contrast great-vs-modhuman on DEV only.

Runs the three pre-registered gates (EXPERIMENT_LOG 2026-06-08), reusing the validated evaluate.py
gauntlet. Does NOT touch the quarantined held-out (that is a separate, PASS-gated step). Writes
results/report_stage1.json and appends the verdict to docs/EXPERIMENT_LOG.md. A clean KILL is a success.
"""
import json
import os
import pandas as pd

from features_lib import PROSODY_COLS
from sequence_features import ORDER_COLS
from lexical_features import LEXICAL_COLS
from evaluate import logreg, _oof_ba, h2_contrast, group_bootstrap_increment_ci

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER_SHUF = [c + "_shuf" for c in ORDER_COLS]
A, B = "great", "modhuman"
SUMMARY_INCREMENT_OVER_SLOP = 0.102   # F2 reference (summary prosody), the bar PRIMARY-3 must clear


def build_dev():
    df = pd.read_parquet(os.path.join(REPO, "results", "features", "features_seq_A.parquet"))
    lex = pd.read_parquet(os.path.join(REPO, "results", "features", "lexical_A_p1c.parquet"))
    split = json.load(open(os.path.join(REPO, "results", "splits", "quarantine_split_p1c.json")))
    lexcols = [c for c in LEXICAL_COLS if c in lex.columns]
    dfp = df.drop(columns=[c for c in ["mattr"] if c in df.columns])   # mattr is lexical's (D11)
    merged = dfp.merge(lex[["passage_id"] + lexcols], on="passage_id", how="inner")
    dev = merged[(merged.passage_id.isin(set(split["dev"]))) & (merged.band == "main")
                 & (merged["class"].isin([A, B]))].reset_index(drop=True)
    return dev, lexcols


def main():
    dev, lexcols = build_dev()
    SUM = [c for c in PROSODY_COLS if c in dev.columns]
    SO, SO_shuf = SUM + ORDER_COLS, SUM + ORDER_SHUF
    y = dev["class"].to_numpy()
    g = dev["source_id"].to_numpy()
    rep = {"contrast": f"{A}_vs_{B}", "n": int(len(dev)), "n_groups": int(pd.unique(g).size),
           "class_counts": dev["class"].value_counts().to_dict()}

    # paired out-of-fold predictions (same folds: cv() is seeded)
    oof_sum, ba_sum = _oof_ba(logreg(), dev[SUM].to_numpy(float), y, g)
    oof_so, ba_so = _oof_ba(logreg(), dev[SO].to_numpy(float), y, g)
    oof_sosh, ba_sosh = _oof_ba(logreg(), dev[SO_shuf].to_numpy(float), y, g)
    oof_or, ba_or = _oof_ba(logreg(), dev[ORDER_COLS].to_numpy(float), y, g)
    oof_osh, ba_osh = _oof_ba(logreg(), dev[ORDER_SHUF].to_numpy(float), y, g)
    rep["ba"] = {"summary": ba_sum, "summary+order": ba_so, "summary+order_shuf": ba_sosh,
                 "order_only": ba_or, "order_only_shuf": ba_osh}

    # PRIMARY-1: lift over the histograms
    lo1, hi1, m1 = group_bootstrap_increment_ci(y, oof_so, oof_sum, g)
    p1 = {"increment": ba_so - ba_sum, "ci": [lo1, hi1], "mean": m1, "pass": bool(lo1 > 0)}

    # PRIMARY-2 (decisive): real order vs shuffled order
    lo2, hi2, m2 = group_bootstrap_increment_ci(y, oof_so, oof_sosh, g)
    lo2b, hi2b, _ = group_bootstrap_increment_ci(y, oof_or, oof_osh, g)
    p2 = {"increment_real_minus_shuffled": ba_so - ba_sosh, "ci": [lo2, hi2], "mean": m2,
          "order_only_real_minus_shuffled": {"increment": ba_or - ba_osh, "ci": [lo2b, hi2b]},
          "pass": bool(lo2 > 0)}

    # PRIMARY-3: orthogonality to lexical (slop-residualized), order-augmented vs summary-only
    h2_so = h2_contrast(dev, SO, lexcols, A, B)
    h2_sum = h2_contrast(dev, SUM, lexcols, A, B)
    rs = h2_so["resid_slop"]
    p3 = {"resid_slop_ba": rs["balanced_acc"], "resid_slop_p": rs["perm_p"], "resid_slop_ci_low": rs["ci_low"],
          "increment_over_slop": h2_so["increment_over_slop"], "increment_ci": h2_so["increment_ci"],
          "summary_only_increment_over_slop": h2_sum["increment_over_slop"],
          "summary_only_resid_slop_ba": h2_sum["resid_slop"]["balanced_acc"],
          "pass": bool(rs["perm_p"] < 0.01 and h2_so["increment_ci"][0] > SUMMARY_INCREMENT_OVER_SLOP)}

    # per-feature order-sensitivity on real DEV (mean |real - shuffled|)
    delta = {c: float((dev[c] - dev[c + "_shuf"]).abs().mean()) for c in ORDER_COLS}
    rep["per_feature_order_sensitivity"] = dict(sorted(((k, round(v, 4)) for k, v in delta.items()),
                                                       key=lambda x: -x[1]))

    rep["PRIMARY_1_lift_over_histograms"] = p1
    rep["PRIMARY_2_order_is_real"] = p2
    rep["PRIMARY_3_orthogonality"] = p3
    rep["gates"] = {"PRIMARY_1": p1["pass"], "PRIMARY_2": p2["pass"], "PRIMARY_3": p3["pass"]}
    rep["verdict"] = "PASS" if all(rep["gates"].values()) else "KILL"

    with open(os.path.join(REPO, "results", "reports", "report_stage1.json"), "w") as f:
        json.dump(rep, f, indent=2)

    lines = [
        f"\n## 2026-06-08 — Stage 1 RESULT (order-aware sequence features): VERDICT {rep['verdict']}",
        f"- great-vs-modhuman DEV n={rep['n']} groups={rep['n_groups']} {rep['class_counts']}",
        f"- ba: summary={ba_sum:.3f} summary+order={ba_so:.3f} summary+order_SHUF={ba_sosh:.3f} | "
        f"order_only={ba_or:.3f} order_only_SHUF={ba_osh:.3f}",
        f"- PRIMARY-1 (lift over histograms): increment={p1['increment']:+.3f} CI[{lo1:.3f},{hi1:.3f}] "
        f"-> {'PASS' if p1['pass'] else 'FAIL'}",
        f"- PRIMARY-2 (order REAL vs SHUFFLED, decisive): increment={p2['increment_real_minus_shuffled']:+.3f} "
        f"CI[{lo2:.3f},{hi2:.3f}] -> {'PASS' if p2['pass'] else 'FAIL'}",
        f"- PRIMARY-3 (orthogonality): resid_slop ba={rs['balanced_acc']:.3f} p={rs['perm_p']:.4f} "
        f"increment_over_slop={h2_so['increment_over_slop']:+.3f} CI{[round(x,3) for x in h2_so['increment_ci']]} "
        f"(summary-only {h2_sum['increment_over_slop']:+.3f}) -> {'PASS' if p3['pass'] else 'FAIL'}",
        f"- per-feature order-sensitivity (mean|real-shuf|): {rep['per_feature_order_sensitivity']}",
        f"- INTERPRETATION: {'order adds real within-human signal beyond histograms+lexical' if rep['verdict']=='PASS' else 'KILL (expected): order-aware hand features add nothing beyond passage histograms on the clean within-human contrast; the ~0.65 ceiling is the SIGNAL/series, not order-blindness (M9 -> do not proceed to CWT/learned-seq on hand features).'}",
    ]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
