#!/usr/bin/env python3
"""Crux judge prep (Claude-judge path; Gemini free tier is quota-blocked for 2.5-pro).

Anonymizes every great/modgreat/modhuman passage so the judge agents are BLIND to class: shuffles them
(seed=0), assigns class-hiding ids p000.., and writes batch files (aid + raw text) for the scoring agents
plus a private mapping.json (aid -> real passage_id) that the agents NEVER see. Anti-cheating: the agents
get only anonymous ids + prose, never the great_/modgreat_/modhuman_ filenames that would leak the label.
"""
import json
import os
import random

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "/tmp/crux_judge"
CLASSES = ["great", "modgreat", "modhuman"]
BATCH = 40


def main():
    os.makedirs(D, exist_ok=True)
    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    m = man[(man.band == "main") & (man["class"].isin(CLASSES))]
    items = []
    for _, r in m.iterrows():
        with open(os.path.join(REPO, "data", "passages", r["class"], r["passage_id"] + ".txt")) as f:
            items.append((r["passage_id"], f.read().strip()))
    random.seed(0)
    random.shuffle(items)

    mapping, batches, cur, bn = {}, [], [], 0
    for i, (pid, text) in enumerate(items):
        aid = f"p{i:03d}"
        mapping[aid] = pid
        cur.append({"aid": aid, "text": text})
        if len(cur) == BATCH or i == len(items) - 1:
            p = os.path.join(D, f"batch_{bn:02d}.json")
            json.dump(cur, open(p, "w"))
            batches.append(p)
            cur, bn = [], bn + 1
    json.dump(mapping, open(os.path.join(D, "mapping.json"), "w"))
    json.dump(batches, open(os.path.join(D, "batches_index.json"), "w"))
    print(f"[prep] {len(items)} passages -> {bn} batches of <= {BATCH}; mapping + index in {D}")


if __name__ == "__main__":
    main()
