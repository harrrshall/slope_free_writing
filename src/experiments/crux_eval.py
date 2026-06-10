#!/usr/bin/env python3
"""Crux test, step 2 - does text-prosody add QUALITY signal BEYOND a frontier LLM judge? Pre-reg + audit
amendment: EXPERIMENT_LOG 2026-06-09.

Binding (per contrast): increment in balanced accuracy of (4-dim judge + prosody) over (judge alone), paired
group-bootstrap CI. Secondary: prosody residualized against the judge (perm test). Reuses the validated
evaluate.py gauntlet so no new statistics code can introduce a bug. Prosody = PROSODY_COLS only (no
lexical/mattr, D11).

Three DEV contrasts (audit-hardened):
  PRIMARY  modgreat vs modhuman  - WITHIN-ERA quality (both modern -> era-clean)
  CONTROL  great    vs modgreat  - ERA control: prosody adding HERE means residual era, not quality
  COMPARE  great    vs modhuman  - original F2 contrast (quality+era confounded)

Per-contrast status: VOID (ba_judge<FLOOR) | ADDS (incr CI low>0) | REDUNDANT (incr CI high<REDUNDANT_UPPER) |
INCONCLUSIVE (underpowered). QUALITY claim requires PRIMARY=ADDS AND CONTROL!=ADDS. Held-out scored ONLY if
PRIMARY ADDS, once, directional only (held-out great/modgreat = 2 groups). Writes results/reports/report_crux.json.
"""
import json
import os

import pandas as pd

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ root on path for library imports
from features_lib import PROSODY_COLS
from evaluate import logreg, _oof_ba, group_bootstrap_increment_ci, lexical_residualized_prosody

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGE_FULL = ["judge_overall", "judge_craft", "judge_coherence", "judge_imagery"]   # strong 4-dim baseline
CLASSES = ["great", "modgreat", "modhuman"]
FLOOR = 0.70             # judge-strength floor: below this the judge is too weak -> VOID
REDUNDANT_UPPER = 0.03   # only call REDUNDANT if the increment CI upper bound is also small
CONTRASTS = [
    ("modgreat", "modhuman", "PRIMARY: within-era quality (era-clean)"),
    ("great", "modgreat", "CONTROL: era (prosody adding here => era not quality)"),
    ("great", "modhuman", "COMPARE: original F2 (quality+era confounded)"),
]


def build(split_key):
    feat = pd.read_parquet(os.path.join(REPO, "results", "features", "features_A_p1c.parquet"))
    judge = pd.read_parquet(os.path.join(REPO, "results", "features", "judge_scores.parquet"))
    split = json.load(open(os.path.join(REPO, "results", "splits", "quarantine_split_p1c.json")))
    feat = feat.drop(columns=[c for c in ["mattr"] if c in feat.columns])   # mattr is lexical's (D11)
    sel = set(split[split_key])
    base = feat[(feat.passage_id.isin(sel)) & (feat.band == "main") & (feat["class"].isin(CLASSES))]
    m = base.merge(judge, on="passage_id", how="inner")
    missing = set(base.passage_id) - set(m.passage_id)
    assert not missing, f"judge scores missing for {len(missing)} {split_key} passages: {sorted(missing)[:8]}"
    assert all(c in m.columns for c in JUDGE_FULL), "judge columns absent"
    return m


def run(m, a, b, judge_cols):
    assert len(judge_cols) >= 1, "empty judge covariate set"
    d = m[m["class"].isin([a, b])].reset_index(drop=True)
    SUM = [c for c in PROSODY_COLS if c in d.columns]
    y = d["class"].to_numpy()
    g = d["source_id"].to_numpy()
    oof_j, ba_j = _oof_ba(logreg(), d[judge_cols].to_numpy(float), y, g)
    oof_jp, ba_jp = _oof_ba(logreg(), d[judge_cols + SUM].to_numpy(float), y, g)
    oof_p, ba_p = _oof_ba(logreg(), d[SUM].to_numpy(float), y, g)
    lo, hi, mean = group_bootstrap_increment_ci(y, oof_jp, oof_j, g)
    rs = lexical_residualized_prosody(d, SUM, judge_cols, perm_n=1000)   # prosody residualized vs the judge
    gc = {cls: int(d[d["class"] == cls].source_id.nunique()) for cls in (a, b)}
    res = {
        "contrast": f"{a}_vs_{b}", "n": int(len(d)), "groups_per_class": gc,
        "ba_judge_only": ba_j, "ba_prosody_only": ba_p, "ba_judge_plus_prosody": ba_jp,
        "increment": ba_jp - ba_j, "increment_ci": [lo, hi], "increment_mean": mean,
        "prosody_resid_judge_ba": rs["balanced_acc"], "prosody_resid_judge_perm_p": rs["perm_p"],
    }
    res["status"] = status(res)
    return res


def status(r):
    if r["ba_judge_only"] < FLOOR:
        return f"VOID (judge too weak: ba_judge={r['ba_judge_only']:.3f}<{FLOOR})"
    if r["increment_ci"][0] > 0:
        return "ADDS (prosody beyond judge)"
    if r["increment_ci"][1] < REDUNDANT_UPPER:
        return "REDUNDANT (judge captures it)"
    return "INCONCLUSIVE (underpowered: CI spans 0 but upper bound large)"


def main():
    dev = build("dev")
    rep = {"pre_reg": "EXPERIMENT_LOG 2026-06-09 (+ amendment)", "floor": FLOOR,
           "redundant_upper": REDUNDANT_UPPER, "judge_baseline": JUDGE_FULL, "DEV": {}}
    for a, b, note in CONTRASTS:
        r = run(dev, a, b, JUDGE_FULL); r["note"] = note
        rep["DEV"][f"{a}_vs_{b}"] = r

    primary = rep["DEV"]["modgreat_vs_modhuman"]
    control = rep["DEV"]["great_vs_modgreat"]
    quality_claim = primary["status"].startswith("ADDS") and not control["status"].startswith("ADDS")
    rep["verdict"] = (
        "PROSODY ADDS QUALITY beyond the judge (primary ADDS, era control does not)" if quality_claim else
        "PRIMARY ADDS but ERA CONTROL also ADDS -> increment is era-confounded, not clean quality"
        if primary["status"].startswith("ADDS") else
        f"NO clean quality signal beyond the judge (primary: {primary['status']})")

    # held-out: only if the DEV primary ADDS; once; directional only
    if primary["status"].startswith("ADDS"):
        held = build("held_out")
        hr = run(held, "modgreat", "modhuman", JUDGE_FULL)
        hr["caveat"] = "DIRECTIONAL ONLY: held-out modgreat=2 groups, not confirmatory power"
        rep["HELDOUT_primary"] = hr
    else:
        rep["HELDOUT_primary"] = {"skipped": "DEV primary did not ADD; held-out NOT touched (M7)"}

    with open(os.path.join(REPO, "results", "reports", "report_crux.json"), "w") as f:
        json.dump(rep, f, indent=2)

    lines = [f"\n## 2026-06-09 - Crux test RESULT (prosody beyond a frontier judge): {rep['verdict']}"]
    for key, r in rep["DEV"].items():
        lines.append(
            f"- {r['note']} [{r['contrast']} n={r['n']} groups={r['groups_per_class']}]: "
            f"ba_judge={r['ba_judge_only']:.3f} ba_pros={r['ba_prosody_only']:.3f} "
            f"ba_judge+pros={r['ba_judge_plus_prosody']:.3f} | increment={r['increment']:+.3f} "
            f"CI[{r['increment_ci'][0]:.3f},{r['increment_ci'][1]:.3f}] | "
            f"resid-vs-judge ba={r['prosody_resid_judge_ba']:.3f} p={r['prosody_resid_judge_perm_p']:.4f} "
            f"-> {r['status']}")
    h = rep["HELDOUT_primary"]
    lines.append(f"- held-out (modgreat-vs-modhuman): " +
                 (h["skipped"] if "skipped" in h else
                  f"increment={h['increment']:+.3f} CI{[round(x,3) for x in h['increment_ci']]} ({h['caveat']})"))
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
