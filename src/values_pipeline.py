#!/usr/bin/env python3
"""VALUES-AXIS experiment (runs on the GPU box, GPU 5 only). Tests whether SPEECH-GROUNDED per-word
prominence (a BERT predictor trained on the Helsinki Prosody Corpus) beats crude dictionary citation-stress
as the prosody front-end, on the three contrasts that matter (modhuman-vs-slop F2, modhuman-vs-gpt4o F6,
great-vs-modhuman quality). Self-contained + robust (writes STATUS/DONE/FAILED). Pre-registered EXPERIMENT_LOG.

Usage: CUDA_VISIBLE_DEVICES=5 python values_pipeline.py [--smoke]
"""
import argparse
import json
import os
import re
import sys
import traceback
from itertools import groupby
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(HERE, "STATUS")
PROM_BLOCK = ["prom_mean", "prom_var", "prom_cv", "prom_runlen_var", "prom_entropy",
              "prom_gap_mean", "prom_gap_var", "prom_ac1"]
SHARED = ["sl_mean", "sl_var", "sl_cv", "ph_mean", "ph_var"]
STRESS = ["stress_bin_entropy", "stress_runlen_var", "gap_mean", "gap_var", "gap_cv",
          "sl_mean", "sl_var", "sl_cv", "ph_mean", "ph_var", "spw_mean", "spw_var", "spw_skew", "oov_rate"]
LEX = ["slopword_density", "slopbigram_density", "sloptrigram_density", "not_x_but_y_rate", "mean_word_len", "fk_grade"]


def mark(s):
    open(STATUS, "w").write(s + "\n")
    print("[status]", s, flush=True)


# ---------- Helsinki corpus parsing ----------
def parse_corpus(path):
    sents, words, labels = [], [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("<file>"):
            if words:
                sents.append((words, labels)); words, labels = [], []
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        w = parts[0]
        try:
            lab = int(parts[1])           # discrete prominence 0/1/2
        except ValueError:
            continue
        words.append(w); labels.append(min(2, max(0, lab)))
    if words:
        sents.append((words, labels))
    return sents


# ---------- train BERT prominence token-classifier ----------
def train_model(smoke):
    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                              DataCollatorForTokenClassification)
    from datasets import Dataset
    mark("loading corpus")
    train = parse_corpus(os.path.join(HERE, "prosody", "data", "train_100.txt"))
    dev = parse_corpus(os.path.join(HERE, "prosody", "data", "dev.txt"))
    if smoke:
        train, dev = train[:500], dev[:100]
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode(batch):
        enc = tok(batch["words"], is_split_into_words=True, truncation=True, max_length=128)
        out_labels = []
        for i, labs in enumerate(batch["labels"]):
            wids = enc.word_ids(i); prev = None; row = []
            for wid in wids:
                if wid is None:
                    row.append(-100)
                elif wid != prev:
                    row.append(labs[wid])
                else:
                    row.append(-100)
                prev = wid
            out_labels.append(row)
        enc["labels"] = out_labels
        return enc

    def to_ds(sents):
        return Dataset.from_dict({"words": [w for w, _ in sents], "labels": [l for _, l in sents]}).map(
            encode, batched=True, remove_columns=["words"])
    train_ds, dev_ds = to_ds(train), to_ds(dev)
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    args = TrainingArguments(output_dir=os.path.join(HERE, "prom_model"), per_device_train_batch_size=32,
                             per_device_eval_batch_size=64, num_train_epochs=1 if smoke else 3,
                             learning_rate=3e-5, logging_steps=200, report_to=[], save_strategy="no",
                             eval_strategy="epoch", fp16=False, bf16=True)
    coll = DataCollatorForTokenClassification(tok)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=coll)
    mark(f"training prominence model ({len(train)} sents, {'smoke' if smoke else 'full'})")
    trainer.train()
    ev = trainer.evaluate()
    mark(f"trained; dev loss={ev.get('eval_loss'):.4f}")
    return tok, model.eval()


# ---------- inference: per-word expected prominence over a passage ----------
def passage_prominence(text, tok, model):
    import torch
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    P = []
    for s in sents:
        words = s.split()
        if not words:
            continue
        enc = tok([words], is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits[0]
        probs = torch.softmax(logits, -1).cpu().numpy()
        exp = probs @ np.array([0.0, 1.0, 2.0])     # expected prominence 0..2 per token
        wids = enc.word_ids(0); prev = None
        for j, wid in enumerate(wids):
            if wid is not None and wid != prev:
                P.append(float(exp[j]))
            prev = wid
    return P


def prom_features(P):
    x = np.asarray(P, float)
    z = {k: 0.0 for k in PROM_BLOCK}
    if len(x) < 3:
        return z
    mean = x.mean()
    z["prom_mean"] = mean; z["prom_var"] = x.var(); z["prom_cv"] = x.std() / mean if mean > 0 else 0.0
    disc = (x > 1.0).astype(int)
    runs = [len(list(g)) for _, g in groupby(disc)]
    z["prom_runlen_var"] = float(np.var(runs)) if runs else 0.0
    hist, _ = np.histogram(x, bins=5, range=(0, 2)); p = hist / hist.sum(); p = p[p > 0]
    z["prom_entropy"] = float(-(p * np.log2(p)).sum()) if p.size else 0.0
    peaks = np.where(disc == 1)[0]; gaps = np.diff(peaks) if len(peaks) >= 2 else np.array([])
    z["prom_gap_mean"] = float(gaps.mean()) if gaps.size else 0.0
    z["prom_gap_var"] = float(gaps.var()) if gaps.size else 0.0
    xc = x - mean; ss = float(xc @ xc)
    z["prom_ac1"] = float(xc[:-1] @ xc[1:] / ss) if ss > 0 else 0.0
    return z


# ---------- contrasts ----------
def run_contrasts(df):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score

    def lr():
        return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])

    def cv():
        return StratifiedGroupKFold(5, shuffle=True, random_state=0)

    def oof(cols, d, y, g):
        return cross_val_predict(lr(), d[cols].to_numpy(float), y, groups=g, cv=cv())

    def gboot_inc(y, oa, ob, g, n=2000, seed=0):
        rng = np.random.default_rng(seed); y, g = np.asarray(y), np.asarray(g)
        uniq = pd.unique(g); idxby = {u: np.where(g == u)[0] for u in uniq}; d = []
        for _ in range(n):
            idx = np.concatenate([idxby[u] for u in rng.choice(uniq, len(uniq), replace=True)])
            if len(set(y[idx])) < 2:
                continue
            d.append(balanced_accuracy_score(y[idx], oa[idx]) - balanced_accuracy_score(y[idx], ob[idx]))
        return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]

    PROM = SHARED + PROM_BLOCK
    out = {}
    for a, b in [("modhuman", "slop"), ("modhuman", "gpt4o"), ("great", "modhuman")]:
        sub = df[df["class"].isin([a, b])].reset_index(drop=True)
        y, g = sub["class"].to_numpy(), sub["source_id"].astype(str).to_numpy()
        o_str = oof(STRESS, sub, y, g); o_prm = oof(PROM, sub, y, g)
        o_both = oof(STRESS + PROM_BLOCK, sub, y, g); o_lex = oof(LEX, sub, y, g)
        ba = balanced_accuracy_score
        out[f"{a}_vs_{b}"] = dict(
            n=int(len(sub)), n_groups=int(pd.unique(g).size),
            stress=round(ba(y, o_str), 3), prominence=round(ba(y, o_prm), 3),
            stress_plus_prom=round(ba(y, o_both), 3), lexical=round(ba(y, o_lex), 3),
            prom_minus_stress=round(ba(y, o_prm) - ba(y, o_str), 3),
            prom_minus_stress_CI=gboot_inc(y, o_prm, o_str, g))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    try:
        mark("START")
        tok, model = train_model(a.smoke)
        old = pd.read_parquet(os.path.join(HERE, "_values_oldfeats.parquet"))
        man = pd.concat([pd.read_csv(os.path.join(HERE, m)) for m in ("manifest.csv", "manifest_gpt4o.csv")],
                        ignore_index=True)
        cls_dir = {"modhuman": "passages/modhuman", "slop": "passages/slop", "great": "passages/great",
                   "gpt4o": "passages_gpt4o/gpt4o"}
        ids = old[["passage_id", "class"]].drop_duplicates()
        if a.smoke:
            ids = ids.groupby("class").head(8)
        mark(f"inferring prominence on {len(ids)} passages")
        rows = []
        for i, (_, r) in enumerate(ids.iterrows()):
            path = os.path.join(HERE, cls_dir[r["class"]], r["passage_id"] + ".txt")
            P = passage_prominence(open(path).read(), tok, model)
            feat = prom_features(P); feat["passage_id"] = r["passage_id"]
            rows.append(feat)
            if (i + 1) % 50 == 0:
                mark(f"inferred {i+1}/{len(ids)}")
        prom = pd.DataFrame(rows)
        df = old.merge(prom, on="passage_id", how="inner")
        mark(f"running contrasts on {len(df)} passages")
        res = run_contrasts(df)
        report = {"smoke": a.smoke, "n": int(len(df)), "contrasts": res}
        json.dump(report, open(os.path.join(HERE, "report_values.json"), "w"), indent=2)

        lines = ["# VALUES-AXIS experiment result (speech-grounded prominence vs dictionary stress)\n",
                 f"(smoke={a.smoke}, n={len(df)} passages)\n",
                 "| contrast | stress(old) | PROMINENCE(new) | stress+prom | lexical | prom-stress (CI) |",
                 "|---|---|---|---|---|---|"]
        for k, v in res.items():
            lines.append(f"| {k} | {v['stress']} | **{v['prominence']}** | {v['stress_plus_prom']} | "
                         f"{v['lexical']} | {v['prom_minus_stress']:+} {v['prom_minus_stress_CI']} |")
        gp = res.get("modhuman_vs_gpt4o", {})
        gq = res.get("great_vs_modhuman", {})
        lines += ["",
                  "## Verdict (pre-registered, honest)",
                  f"- F6 GPT-4o detection: stress={gp.get('stress')} -> prominence={gp.get('prominence')} "
                  f"(prom-stress {gp.get('prom_minus_stress')}, CI {gp.get('prom_minus_stress_CI')})",
                  f"- Quality (great-vs-modhuman): stress={gq.get('stress')} -> prominence={gq.get('prominence')} "
                  f"(prom-stress {gq.get('prom_minus_stress')}, CI {gq.get('prom_minus_stress_CI')})",
                  "- EXTRACTION WAS THE BOTTLENECK iff prominence beats stress with CI lower bound > 0 on a "
                  "contrast that was weak (esp. GPT-4o or quality). Otherwise the signal is genuinely weak."]
        open(os.path.join(HERE, "report.md"), "w").write("\n".join(lines) + "\n")
        mark("DONE")
        print("\n".join(lines))
    except Exception:
        tb = traceback.format_exc()
        open(os.path.join(HERE, "FAILED"), "w").write(tb)
        mark("FAILED")
        print(tb, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
