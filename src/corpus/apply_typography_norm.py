#!/usr/bin/env python3
"""Phase-1c: apply normalize_typography UNIFORMLY to every passage IN PLACE.

Integration point (recommended): re-clean the EXISTING passage .txt files (same
passage_ids) rather than normalizing at feature-extraction time. Rationale:
  * quarantine_split_p1.json and the held-out are keyed by passage_id only, so an
    in-place rewrite leaves the carry-forward quarantine and the once-scored held-out
    split completely undisturbed (no ids change, no rows added/removed).
  * every downstream consumer (extract_features.py, extract_lexical.py, the
    typography-only diagnostic) reads from data/passages/, so one rewrite normalizes
    the whole pipeline; there is no risk of "normalized here, raw there" drift.
  * the manifest's char_len is the only stale column; it is refreshed here.

Safety: writes a one-time backup tree data/passages_raw/ (skipped if it already
exists, so re-running is idempotent and never clobbers the pristine originals).

After this runs, re-extract:
  .venv/bin/python extract_features.py --out features_A_p1.parquet
  .venv/bin/python extract_lexical.py  --out lexical_A_p1.parquet
then re-run evaluate.py on the (unchanged) quarantine_split_p1.json.
"""
import os, shutil, sys
import pandas as pd
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ root on path for library imports
from typography import normalize_typography

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS = os.path.join(REPO, "data", "passages")
RAW = os.path.join(REPO, "data", "passages_raw")


def main():
    dry = "--dry-run" in sys.argv
    if not os.path.isdir(PASS):
        sys.exit(f"[typo] no passages dir at {PASS}")

    # 1) one-time pristine backup (idempotent).
    if not os.path.isdir(RAW):
        if dry:
            print(f"[typo] (dry-run) would back up {PASS} -> {RAW}")
        else:
            shutil.copytree(PASS, RAW)
            print(f"[typo] backed up pristine passages -> {RAW}")
    else:
        print(f"[typo] backup {RAW} already exists; leaving it untouched")

    # 2) rewrite every .txt in place; ALWAYS read the pristine source (RAW if present)
    #    so re-running is deterministic and never double-normalizes.
    src_root = RAW if os.path.isdir(RAW) and not dry else PASS
    man = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
    changed, total, char_updates = 0, 0, {}
    for _, r in man.iterrows():
        rel = os.path.join(r["class"], r["passage_id"] + ".txt")
        src = os.path.join(src_root, rel)
        dst = os.path.join(PASS, rel)
        if not os.path.exists(src):
            print(f"[typo] WARN missing {src}")
            continue
        raw = open(src, encoding="utf-8").read()
        out = normalize_typography(raw)
        total += 1
        if out != raw:
            changed += 1
        char_updates[r["passage_id"]] = len(out)
        if not dry:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(out)
    print(f"[typo] normalized {changed}/{total} passages "
          f"({'DRY-RUN, nothing written' if dry else 'written in place'})")

    # 3) refresh manifest char_len (only column made stale by re-cleaning).
    if not dry:
        man["char_len"] = man["passage_id"].map(char_updates).fillna(man["char_len"]).astype(int)
        man.to_csv(os.path.join(REPO, "data", "manifest.csv"), index=False)
        print("[typo] refreshed manifest.csv char_len")
    print("[typo] DONE. Re-extract features then re-run evaluate.py on the unchanged split.")


if __name__ == "__main__":
    main()
