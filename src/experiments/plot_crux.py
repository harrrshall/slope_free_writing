#!/usr/bin/env python3
"""Render the crux result (F8) as a two-panel figure from results/reports/report_crux.json.
Panel A: ranking writing quality (within-era) - the judge is near-perfect, prosody adds nothing.
Panel B: the era control - the judge is at chance but prosody is not, so prosody tracks era, not quality.
Writes figures/crux_result.png. No em-dashes in any text (project writing rule).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
C_JUDGE, C_PROS, C_BOTH = "#4ea1ff", "#ff7b72", "#56d364"


def main():
    r = json.load(open(os.path.join(REPO, "results", "reports", "report_crux.json")))["DEV"]
    q = r["modgreat_vs_modhuman"]      # within-era quality
    e = r["great_vs_modgreat"]         # era control

    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "font.size": 12})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # Panel A: quality
    labA = ["Judge\nalone", "Prosody\nalone", "Judge +\nProsody"]
    valA = [q["ba_judge_only"], q["ba_prosody_only"], q["ba_judge_plus_prosody"]]
    barsA = axA.bar(labA, valA, color=[C_JUDGE, C_PROS, C_BOTH], width=0.62)
    axA.set_title("Ranking writing quality\n(modern great vs amateur, same era)", color=FG, fontsize=13, pad=12)
    axA.text(0.5, 0.5, "adding prosody to the judge: +0.00", transform=axA.transAxes, ha="center",
             color="#8b949e", fontsize=11, style="italic")

    # Panel B: era control
    labB = ["Judge\nalone", "Prosody\nalone"]
    valB = [e["ba_judge_only"], e["ba_prosody_only"]]
    axB.bar(labB, valB, color=[C_JUDGE, C_PROS], width=0.5)
    axB.set_title("Telling eras apart\n(old great vs modern great, same quality)", color=FG, fontsize=13, pad=12)
    axB.text(0.5, 0.88, "the quality judge is at chance,\nbut prosody is not: it tracks era, not quality",
             transform=axB.transAxes, ha="center", color="#8b949e", fontsize=11, style="italic")

    for ax, vals, labs in ((axA, valA, labA), (axB, valB, labB)):
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("balanced accuracy")
        ax.axhline(0.5, color="#8b949e", ls="--", lw=1)
        ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", ha="left", color="#8b949e", fontsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        for x, v in enumerate(vals):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", color=FG, fontsize=12, fontweight="bold")

    fig.suptitle("A strong judge already captures writing quality. Prosody adds nothing; what it still detects is era.",
                 color=FG, fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(REPO, "figures", "crux_result.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
