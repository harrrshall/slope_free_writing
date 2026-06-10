#!/usr/bin/env python3
"""Apply the FROZEN Phase-1b corrected kill gate to report_p1.json (no threshold changes post-hoc).

H2 (orthogonality) binds on the SLOP-baseline residual for GREAT-vs-MODHUMAN (both human,
non-circular). H3 binds on modhuman-vs-slop prosody-only. Full-lexical residuals + great-vs-slop
are reported as diagnostics.
"""
import argparse, json, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    r = json.load(open(os.path.join(REPO, args.report)))
    h2, h3 = r["h2"], r["h3"]

    gm = h2.get("great_vs_modhuman")          # BINDING H2b.1
    if gm is None:
        print("great_vs_modhuman absent (need the modern corpus) -> cannot evaluate corrected H2")
        return "INCOMPLETE"
    rs = gm["resid_slop"]
    h2_proceed = (rs["ci_low"] > 0.50 and rs["perm_p"] < 0.01)
    h2_kill = (rs["ci_low"] <= 0.50 or rs["perm_p"] >= 0.05)

    ms = h3.get("modhuman_vs_slop")           # BINDING H3-A
    h3_proceed = ms and (ms["balanced_acc"] >= 0.65 and ms["ci_low"] > 0.50 and ms["perm_p"] < 0.01)
    h3_kill = ms and (ms["balanced_acc"] <= 0.55 or ms["ci_low"] <= 0.50 or ms["perm_p"] >= 0.05)

    print(f"=== PHASE-1b CORRECTED GATE on {args.report} ===")
    print("-- H2 (orthogonality, binding = great-vs-modhuman vs SLOP baseline) --")
    print(f"  [{'PASS' if h2_proceed else 'FAIL'}] great-vs-modhuman resid-vs-SLOP: ba={rs['balanced_acc']:.3f} "
          f"CI[{rs['ci_low']:.3f},{rs['ci_high']:.3f}] perm_p={rs['perm_p']:.4f}  "
          f"(prosody_only={gm['prosody_only']:.3f} slop_baseline={gm['lexical_slop_only']:.3f} "
          f"increment_over_slop={gm['increment_over_slop']:+.3f})")
    for k in ("great_vs_slop", "modhuman_vs_slop"):
        v = h2.get(k)
        if v:
            print(f"  [report] {k} resid-vs-SLOP ba={v['resid_slop']['balanced_acc']:.3f} "
                  f"CI_low={v['resid_slop']['ci_low']:.3f} p={v['resid_slop']['perm_p']:.4f} | "
                  f"resid-vs-FULL(diag) ba={v['resid_full']['balanced_acc']:.3f} CI_low={v['resid_full']['ci_low']:.3f}")
    print("-- H3 (genre+era, binding = modhuman-vs-slop) --")
    if ms:
        print(f"  [{'PASS' if h3_proceed else 'FAIL'}] modhuman-vs-slop ba={ms['balanced_acc']:.3f} "
              f"CI[{ms['ci_low']:.3f},{ms['ci_high']:.3f}] perm_p={ms['perm_p']:.4f}")
        for k in ("great_vs_modhuman", "great_vs_modhuman_LEXICALonly", "great_vs_modgreat_full",
                  "modgreat_vs_slop_subsent_noHF", "modgreat_vs_great_subsent_noHF"):
            if h3.get(k):
                print(f"  [report] {k} ba={h3[k]['balanced_acc']:.3f}"
                      + (f" p={h3[k]['perm_p']:.4f}" if "perm_p" in h3[k] else ""))

    if h2_kill or h3_kill:
        verdict = "KILL"
    elif h2_proceed and h3_proceed:
        verdict = "PROCEED (pending held-out confirmation H2b.2)"
    else:
        verdict = "NO-GO (a PROCEED condition failed; not a hard kill)"
    print(f"\nVERDICT: {verdict}")
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as f:
        f.write(f"\n- PHASE-1b GATE {args.report}: VERDICT={verdict} "
                f"(H2b.1 great-vs-modhuman resid-vs-SLOP ba={rs['balanced_acc']:.3f} CI_low={rs['ci_low']:.3f} p={rs['perm_p']:.4f}; "
                + (f"H3-A modhuman-vs-slop ba={ms['balanced_acc']:.3f} CI_low={ms['ci_low']:.3f} p={ms['perm_p']:.4f}" if ms else "modhuman absent") + ")\n")
    return verdict


if __name__ == "__main__":
    main()
