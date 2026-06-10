#!/usr/bin/env python3
"""Sequentially fetch the N smallest ZuCo 2.0 NR subjects: download -> parse to per-word CSV -> delete
the .mat. Peak disk = one .mat at a time. Idempotent: skips subjects already parsed.

Usage: python src/zuco/zuco_fetch.py <N>   (reads /tmp/zuco/subject_urls.json)
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(n):
    subs = json.load(open("/tmp/zuco/subject_urls.json"))
    order = sorted(subs.items(), key=lambda x: x[1]["mb"])[:n]
    for name, v in order:
        subj = name.replace("results", "").replace("_NR.mat", "")
        out = os.path.join(REPO, "data", "zuco", f"words_{subj}.csv")
        if os.path.exists(out):
            print(f"skip {subj} (already parsed)", flush=True)
            continue
        mat = os.path.join(REPO, "data", "zuco", f"_tmp_{subj}.mat")
        print(f"downloading {subj} ({v['mb']}MB)...", flush=True)
        subprocess.run(["curl", "-sL", "--max-time", "1200", v["url"], "-o", mat], check=True)
        subprocess.run([sys.executable, os.path.join(REPO, "src", "zuco", "zuco_parse.py"), mat, out], check=True)
        os.remove(mat)
    print("DONE fetch+parse", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
