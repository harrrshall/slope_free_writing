#!/usr/bin/env python3
"""Group-aware quarantine split (~20% held-out, no source crosses dev/held-out).

Group = source_id (book / reuters-category / model). Held-out is CONFIRMATORY ONLY
and is scored exactly once later. Asserts the DEV set can do a non-degenerate 5-fold
StratifiedGroupKFold (every fold has all 3 classes). Appends held-out n/class to the
pre-registered EXPERIMENT_LOG.md block.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = os.path.dirname(os.path.abspath(__file__))


def carry_forward(prev_path, out_path):
    """Keep Phase-0 held-out/dev membership FIXED; assign NEW classes group-aware. Asserts the p1
    held-out is a superset of the Phase-0 held-out and that no source_id crosses dev/held-out."""
    man = pd.read_csv(os.path.join(REPO, "manifest.csv"))
    main_df = man[man.band == "main"].reset_index(drop=True)
    prev = json.load(open(os.path.join(REPO, prev_path)))
    prev_held, prev_dev = set(prev["held_out"]), set(prev["dev"])
    known = prev_held | prev_dev
    held, dev = set(prev_held), set(prev_dev)
    rng = np.random.default_rng(0)
    new = main_df[~main_df.passage_id.isin(known)]
    for cls, grp in new.groupby("class"):
        srcs = list(pd.unique(grp["source_id"]))
        srcs = [srcs[i] for i in rng.permutation(len(srcs))]
        target = int(round(0.20 * len(grp)))
        h = 0
        for s in srcs:
            ids = grp[grp.source_id == s]["passage_id"].tolist()
            if h < target:
                held.update(ids); h += len(ids)
            else:
                dev.update(ids)
        print(f"[carry] {cls}: held {h}/{len(grp)} ({pd.unique(grp['source_id']).size} groups)")

    md = main_df.set_index("passage_id")
    split = {"held_out": sorted(held), "dev": sorted(dev)}
    json.dump(split, open(os.path.join(REPO, out_path), "w"), indent=2)
    assert prev_held.issubset(held), "p1 held-out must be a superset of Phase-0 held-out"
    assert prev_dev.issubset(dev), "p1 dev must be a superset of Phase-0 dev"
    hsrc = set(md.loc[[i for i in held if i in md.index], "source_id"])
    dsrc = set(md.loc[[i for i in dev if i in md.index], "source_id"])
    assert not (hsrc & dsrc), f"group overlap dev/held: {hsrc & dsrc}"
    held_df = md.loc[[i for i in held if i in md.index]]
    counts = held_df["class"].value_counts().to_dict()
    print(f"[carry] wrote {out_path}: held-out n={len(held)} by class={counts}; dev n={len(dev)}")
    with open(os.path.join(REPO, "EXPERIMENT_LOG.md"), "a") as f:
        f.write(f"\n- P1 carry-forward split {out_path}: held-out by class={counts}, "
                f"total held={len(held)}, dev={len(dev)}; group overlap=empty; superset of Phase-0 held-out=OK.\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry-forward", default=None, dest="carry")
    ap.add_argument("--out", default="quarantine_split.json")
    args = ap.parse_args()
    if args.carry:
        return carry_forward(args.carry, args.out)
    man = pd.read_csv(os.path.join(REPO, "manifest.csv"))
    main_df = man[man.band == "main"].reset_index(drop=True)
    y = main_df["class"].to_numpy()
    groups = main_df["source_id"].to_numpy()
    ids = main_df["passage_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    chosen = None
    for dev_idx, held_idx in sgkf.split(main_df, y, groups):
        held_classes = set(y[held_idx])
        if {"great", "flat", "slop"}.issubset(held_classes):
            # dev must also support a non-degenerate 5-fold
            dev_y, dev_g = y[dev_idx], groups[dev_idx]
            ok = True
            try:
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
                        np.zeros(len(dev_idx)), dev_y, dev_g):
                    if len(set(dev_y[te])) < 3:
                        ok = False; break
            except ValueError:
                ok = False
            if ok:
                chosen = (dev_idx, held_idx); break
    if chosen is None:
        raise SystemExit("[split] FATAL: could not find a group-aware split with all classes held out")

    dev_idx, held_idx = chosen
    split = {"held_out": ids[held_idx].tolist(), "dev": ids[dev_idx].tolist()}
    with open(os.path.join(REPO, "quarantine_split.json"), "w") as f:
        json.dump(split, f, indent=2)

    held = main_df.iloc[held_idx]
    dev = main_df.iloc[dev_idx]
    held_counts = held["class"].value_counts().to_dict()
    print(f"[split] held-out n={len(held_idx)} by class={held_counts}")
    print(f"[split] dev n={len(dev_idx)} by class={dev['class'].value_counts().to_dict()}")
    # confirm no group leakage
    leak = set(held["source_id"]) & set(dev["source_id"])
    print(f"[split] group overlap dev/held (must be empty): {leak}")
    assert not leak, "group leakage between dev and held-out"

    # fill the pre-registered held-out n into the log
    line = (f"\n- Held-out n per class (filled at P0.3): "
            f"great={held_counts.get('great',0)}, flat={held_counts.get('flat',0)}, "
            f"slop={held_counts.get('slop',0)}; total held-out={len(held_idx)}, dev={len(dev_idx)}.\n")
    with open(os.path.join(REPO, "EXPERIMENT_LOG.md"), "a") as f:
        f.write(line)
    print("[split] wrote quarantine_split.json and logged held-out n/class")


if __name__ == "__main__":
    main()
