#!/usr/bin/env python3
"""Parse ONE ZuCo 2.0 NR .mat (MATLAB v7.3 / HDF5) into a tiny per-word CSV.

Keeps only eye-tracking (FFD, GD, GPT, TRT, nFixations) + one theta-band EEG aggregate per word,
and the sentence text. rawEEG/rawET are NOT read, so memory stays low and the 1-2GB source can be
deleted right after. Field logic mirrors the official ZuCo data_loading_helpers.extract_word_level_data
(but uses h5py `[()]` since `.value` was removed in modern h5py).

Usage: python src/zuco/zuco_parse.py <path_to_results.mat> <out.csv>
"""
import os
import re
import sys
import numpy as np
import h5py
import pandas as pd


def _mstr(ds):
    """MATLAB uint16 char array -> python str."""
    return "".join(chr(int(c)) for c in np.array(ds).ravel())


def _scalar(f, ref):
    """Dereference an object ref to a scalar float; NaN if missing/empty (word not fixated)."""
    try:
        v = np.asarray(f[ref][()], dtype=float).ravel()
        return float(v[0]) if v.size else np.nan
    except Exception:
        return np.nan


def _theta_trt(f, wg, w):
    """Mean over EEG channels of the theta-band feature during total reading time (engagement proxy)."""
    try:
        v = np.asarray(f[wg["TRT_t1"][w][0]][()], dtype=float).ravel()
        return float(np.nanmean(v)) if v.size else np.nan
    except Exception:
        return np.nan


def parse(mat_path, out_csv):
    subj = os.path.basename(mat_path).replace("results", "").replace("_NR.mat", "")
    f = h5py.File(mat_path, "r")
    sd = f["sentenceData"]
    content, word = sd["content"], sd["word"]
    rows = []
    for s in range(content.shape[0]):
        try:
            sent = _mstr(f[content[s][0]])
            wg = f[word[s][0]]
            if "FFD" not in wg:               # sentence has no word-level ET/EEG
                continue
            wi = 0
            for w in range(wg["content"].shape[0]):
                try:
                    ws = _mstr(f[wg["content"][w][0]])
                    if not re.search(r"[A-Za-z0-9]", ws):
                        continue
                    rows.append(dict(
                        subject=subj, sent_idx=s, word_idx=wi, word=ws,
                        FFD=_scalar(f, wg["FFD"][w][0]), GD=_scalar(f, wg["GD"][w][0]),
                        GPT=_scalar(f, wg["GPT"][w][0]), TRT=_scalar(f, wg["TRT"][w][0]),
                        nFix=_scalar(f, wg["nFixations"][w][0]), theta_trt=_theta_trt(f, wg, w),
                        sentence=sent))
                    wi += 1
                except Exception:
                    continue
        except Exception:
            continue
    f.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    frac = df["FFD"].notna().mean() if len(df) else 0.0
    print(f"[zuco_parse] {subj}: {len(df)} words, {df['sent_idx'].nunique()} sentences, "
          f"FFD non-null {frac:.2f} -> {out_csv}", flush=True)
    return len(df)


if __name__ == "__main__":
    parse(sys.argv[1], sys.argv[2])
