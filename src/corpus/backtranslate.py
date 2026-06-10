#!/usr/bin/env python3
"""Back-translate passages EN->DE->EN (Helsinki-NLP opus-mt) as an INDEPENDENT, non-LLM paraphraser
(Track A generality, Z2). A passage is translated whole when short enough, else split into 2 at a sentence
boundary so the MT can restructure within each part (fairer than sentence-by-sentence, which would preserve
the sentence-length pattern by construction). Deterministic (beam search)."""
import argparse
import os
import pandas as pd
import spacy
import torch
from transformers import MarianMTModel, MarianTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name):
    return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).eval()


def split_parts(text, nlp, max_words=300):
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sents:
        return []
    if sum(len(s.split()) for s in sents) <= max_words:
        return [" ".join(sents)]
    mid = len(sents) // 2
    return [" ".join(sents[:mid]), " ".join(sents[mid:])]


def translate(tok, mod, texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            g = mod.generate(**b, max_length=512, num_beams=1)   # greedy: ~4x faster, fine for paraphrase
        out.extend(tok.batch_decode(g, skip_special_tokens=True))
        print(f"[bt]   {min(i + bs, len(texts))}/{len(texts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True, help="comma list, e.g. modhuman,slop")
    ap.add_argument("--src-dir", default="data/passages")
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    a = ap.parse_args()
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    ten, men = load("Helsinki-NLP/opus-mt-en-de")
    tde, mde = load("Helsinki-NLP/opus-mt-de-en")
    man = pd.read_csv(os.path.join(REPO, a.manifest))
    rows = man[(man.band == "main") & (man["class"].isin(a.classes.split(",")))]

    pmap = {}
    for _, r in rows.iterrows():
        path = os.path.join(REPO, a.src_dir, r["class"], r["passage_id"] + ".txt")
        if not os.path.exists(path):
            continue
        parts = split_parts(open(path).read().strip(), nlp)
        if parts:
            pmap[(r["passage_id"], r["class"])] = parts

    flat, idx = [], []
    for k, parts in pmap.items():
        for j, p in enumerate(parts):
            flat.append(p); idx.append((k, j))
    print(f"[bt] {len(pmap)} passages -> {len(flat)} parts; EN->DE ...", flush=True)
    de = translate(ten, men, flat)
    print("[bt] DE->EN ...", flush=True)
    en = translate(tde, mde, de)

    res = {}
    for (k, j), bt in zip(idx, en):
        res.setdefault(k, {})[j] = bt
    for (pid, cls), parts in res.items():
        d = os.path.join(REPO, a.dst_dir, cls)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, pid + ".txt"), "w").write(" ".join(parts[j] for j in sorted(parts)))
    print(f"[bt] wrote {len(res)} back-translated passages to {a.dst_dir}", flush=True)


if __name__ == "__main__":
    main()
