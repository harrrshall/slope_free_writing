#!/usr/bin/env python3
"""Apply the PRE-REGISTERED frozen kill criterion to a report_*.json.

Prints PROCEED / KILL / DANGER / NO-GO with a per-condition table. Does not change
any threshold (M7). NO-GO = ba>=0.65 but a non-kill PROCEED condition failed -> per the
frozen 'PROCEED iff ALL', we do not advance (functionally a non-validation).
"""
import argparse, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    r = json.load(open(os.path.join(REPO, args.report)))

    bl = r["binary"]["logreg"]
    ba, ci_low, p = bl["balanced_acc"], bl["ci_low"], bl["perm_p"]
    best_base = max(r["baselines"]["length_only_ba"], r["baselines"]["mattr_only_ba"])
    delta = r["prosody_minus_best_baseline"]
    resid_low = r["residualized"]["ci_low"]
    era = r["era_balanced"]["binary_ba"]

    proceed = {
        "1 binary ba>=0.65 & CI_low>0.50": (ba >= 0.65 and ci_low > 0.50),
        "2 perm_p<0.01": (p < 0.01),
        "3 rhythm group in top3": bool(r["rhythm_group_in_top3"]),
        "4 prosody - best_baseline >= 0.05": (delta >= 0.05),
        "5 residualized CI_low>0.50": (resid_low > 0.50),
        "6 era-balanced ba>=0.60": (era >= 0.60),
    }
    kill = {
        "binary ba<=0.55 or CI includes 0.50": (ba <= 0.55 or ci_low <= 0.50),
        "perm_p>=0.05": (p >= 0.05),
        "residualized collapses (CI_low<=0.50)": (resid_low <= 0.50),
        "signal reproduced by length/MATTR baseline": (best_base >= ba),
    }

    if any(kill.values()):
        verdict = "KILL"
    elif all(proceed.values()):
        verdict = "PROCEED"
    elif 0.55 < ba < 0.65:
        verdict = "DANGER (one rescue round permitted)"
    else:
        verdict = "NO-GO (fails a PROCEED condition; not a listed hard-kill -> do not advance)"

    print(f"=== KILL-CRITERION GATE on {args.report} ===")
    print(f"binary great-vs-flat ba={ba:.3f} CI_low={ci_low:.3f} perm_p={p:.4f}")
    print(f"best baseline={best_base:.3f}  prosody-baseline delta={delta:+.3f}")
    print(f"residualized CI_low={resid_low:.3f}  era-balanced ba={era:.3f}")
    print("-- PROCEED conditions --")
    for k, v in proceed.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("-- KILL triggers --")
    for k, v in kill.items():
        print(f"  [{'TRIGGERED' if v else 'ok'}] {k}")
    print(f"\nVERDICT: {verdict}")

    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(f"\n- GATE {args.report}: VERDICT={verdict} "
                f"(ba={ba:.3f}, CI_low={ci_low:.3f}, perm_p={p:.4f}, delta={delta:+.3f}, "
                f"resid_low={resid_low:.3f}, era={era:.3f})\n")
    return verdict


if __name__ == "__main__":
    main()
