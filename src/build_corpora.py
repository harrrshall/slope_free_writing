#!/usr/bin/env python3
"""Phase 0 corpus builder.

Builds three length-matched piles (GREAT / FLAT / SLOP) at N=80/class plus a
modern-human era-sensitivity band (euclaise), with every integrity fix from the
reviewed protocol:

  * UNIFORM sentence-boundary chunking for ALL piles (accumulate whole spaCy
    sentences to a [150,400]-word budget; never split a sentence) so the chunking
    method is identical per class and cannot inject a class-correlated
    sentence-length-variance artifact.
  * Gutenberg: head-12% / tail-5% paragraph trim + heading/all-caps filter +
    underscore-italics stripped (strip_headers leaves intros/TOC/[Illustration]).
  * Reuters: drop the leading ALL-CAPS headline line; sample across categories.
  * AI-slop: filter language=='en', de-dup on story_text, select the bottom
    quartile of overall_score PER MODEL (stratified), then per-model caps; log the
    final model distribution (no source > ~15% of its class).
  * euclaise: detokenized with sacremoses; built as a separate band ONLY (never in
    the held-out, never the M4 'real human prose' anchor).

Deterministic (numpy default_rng(0)). Writes data/passages/<class>/*.txt and
manifest.csv. Real human prose (Gutenberg + Reuters) satisfies M4.
"""
import os, re, csv, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LO, HI = 150, 400
N_PER_CLASS = 80
PER_SOURCE_CAP = 12          # ~15% of 80
N_BINS = 5
SEED = 0
rng = np.random.default_rng(SEED)

# Acknowledged-great, pre-1928 literary fiction (era confound accepted: user chose Gutenberg-only).
BOOK_IDS = {
    1342: "Austen_PridePrejudice", 1400: "Dickens_GreatExpectations",
    145: "Eliot_Middlemarch", 541: "Wharton_AgeOfInnocence",
    219: "Conrad_HeartOfDarkness", 76: "Twain_HuckFinn",
    120: "Stevenson_TreasureIsland", 2701: "Melville_MobyDick",
    1260: "Bronte_JaneEyre", 174: "Wilde_DorianGray",
    84: "Shelley_Frankenstein", 158: "Austen_Emma",
}

print("[build] loading spaCy en_core_web_sm ...", flush=True)
import spacy
NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
NLP.max_length = 2_000_000


def wc(s: str) -> int:
    return len(s.split())


def chunk_text(text, lo=LO, hi=HI, char_cap=120_000):
    """Uniform chunker with a RANDOMIZED per-chunk target length in [lo,hi].

    Packing continuous text to a fixed budget piles every chunk near the cap, which
    made GREAT/SLOP ~380w while short Reuters docs stayed ~273w (an M5 length confound).
    Drawing a fresh target per chunk spreads lengths uniformly so all piles populate
    every word-count bin and can be length-matched at the sampling stage.
    """
    text = re.sub(r"\s+", " ", text).strip()[:char_cap]
    if not text:
        return []
    doc = NLP(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks, buf, n = [], [], 0
    target = int(rng.integers(lo, hi + 1))
    for s in sents:
        sw = len(s.split())
        if not buf and sw > hi:            # a single oversized sentence: cannot fit
            continue
        buf.append(s); n += sw
        if n >= target or n >= hi:
            if lo <= n <= hi:
                chunks.append(" ".join(buf))
            buf, n, target = [], 0, int(rng.integers(lo, hi + 1))
    if lo <= n <= hi:
        chunks.append(" ".join(buf))
    return chunks


# ---------------------------------------------------------------- GREAT
def clean_gutenberg(raw):
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    n = len(paras)
    paras = paras[int(n * 0.12): int(n * 0.95)]              # drop front/back matter
    bad = re.compile(r"^(CHAPTER|VOLUME|BOOK|THE END|CONTENTS|ILLUSTRATION|PREFACE|INTRODUCTION)\b", re.I)
    out = []
    for p in paras:
        if p.isupper():                                       # all-caps heading
            continue
        if bad.match(p):
            continue
        if re.match(r"^\[?Illustration", p, re.I):
            continue
        p = p.replace("_", "")                                # italics markers
        out.append(p)
    return " ".join(out)


# Phase-1 modgreat: acknowledged-great MODERN fiction now US-public-domain (1922-1930), verified
# fetchable. EXPLORATORY/non-gating (D13); Hemingway/Faulkner are the F1 driver -> circularity guard
# applied downstream (sub-sentence-only, held out in the gated analysis).
MODERN_GREAT_IDS = {
    64317: "Fitzgerald_Gatsby_1925", 75201: "Hemingway_FarewellToArms_1929",
    67138: "Hemingway_SunAlsoRises_1926", 75170: "Faulkner_SoundAndFury_1929",
    77600: "Hammett_MalteseFalcon_1930", 75011: "Remarque_AllQuietWesternFront_1929",
    67979: "Montgomery_BlueCastle_1926", 16389: "vonArnim_EnchantedApril_1922",
}


def build_great(ids=BOOK_IDS, era="pre1928", tag="great"):
    import gutenbergpy.textget as tg
    cands = []
    for bid, name in ids.items():
        try:
            raw = tg.strip_headers(tg.get_text_by_id(bid)).decode("utf-8", "ignore")
        except Exception as e:
            print(f"[{tag}] SKIP book {bid} ({name}): {e}", flush=True)
            continue
        text = clean_gutenberg(raw)
        if len(text.split()) < 1000:                          # transcription absent/empty
            print(f"[{tag}] SKIP book {bid} ({name}): too short ({len(text.split())}w)", flush=True)
            continue
        chunks = chunk_text(text)
        for c in chunks[:60]:
            cands.append(dict(text=c, source_id=name, wc=wc(c), era=era))
        print(f"[{tag}] {name} (#{bid}) -> {min(len(chunks),60)} pooled", flush=True)
    return cands


# ---------------------------------------------------------------- MODHUMAN (Fan-2018, D10)
import html
_WP_TAG = re.compile(r"\[[A-Za-z]{2,4}\]")
_URL = re.compile(r"http\S+")
_MDLINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def clean_reddit(t):
    """Strip forum/markdown register so modhuman typography is not a NEW confound vs Gutenberg (M5)."""
    t = html.unescape(t)
    t = _URL.sub("", t)
    t = _MDLINK.sub(r"\1", t)
    t = _WP_TAG.sub("", t)                                     # [WP] [CW] [EU] [TT] [IP] ...
    t = t.replace("**", "").replace("__", "").replace("~~", "").replace("`", "")
    t = re.sub(r"(?m)^\s*[>#]+", "", t)                        # blockquote / header markers
    t = re.sub(r"(?i)\bedit\s*:.*", "", t)                     # 'EDIT:' addenda
    t = t.replace("*", "").replace("_", "")
    return re.sub(r"\s+", " ", t).strip()


def build_modhuman(target_pool=500, per_story=3):
    from datasets import load_dataset
    from sacremoses import MosesDetokenizer
    md = MosesDetokenizer(lang="en")
    print("[modhuman] loading frozen Fan-2018 WritingPrompts (euclaise/writingprompts) ...", flush=True)
    ds = load_dataset("euclaise/writingprompts", split="train")
    field = "story" if "story" in ds.column_names else ds.column_names[-1]
    cands = []
    for i, row in enumerate(ds):
        if len(cands) >= target_pool:
            break
        txt = row[field]
        if not txt:
            continue
        txt = clean_reddit(md.detokenize(str(txt).split()))    # detokenize THEN register-clean
        for c in chunk_text(txt)[:per_story]:                  # pool chunks[:k], not opening-only
            cands.append(dict(text=c, source_id=f"wp_{i}", wc=wc(c), era="modern_human"))
    return cands


# ---------------------------------------------------------------- FLAT
def build_flat():
    import nltk
    from nltk.corpus import reuters
    cats = sorted(reuters.categories())
    rng.shuffle(cats)
    cands = []
    for cat in cats:
        if len([c for c in cands]) > 320:
            break
        fids = [f for f in reuters.fileids(cat) if f.startswith("training/")][:25]
        pooled = 0
        for fid in fids:
            if pooled >= 25:
                break
            lines = [l for l in reuters.raw(fid).splitlines() if l.strip()]
            if not lines:
                continue
            body = " ".join(lines[1:]) if lines[0].isupper() else " ".join(lines)
            for c in chunk_text(body):
                cands.append(dict(text=c, source_id=f"cat:{cat}", wc=wc(c), era="1987"))
                pooled += 1
        if pooled:
            print(f"[flat] cat:{cat} -> {pooled} pooled", flush=True)
    return cands


# ---------------------------------------------------------------- SLOP
def build_slop():
    from datasets import load_dataset
    print("[slop] loading lars1234/story_writing_benchmark (config=average) ...", flush=True)
    ds = load_dataset("lars1234/story_writing_benchmark", "average", split="train")
    import pandas as pd
    df = ds.to_pandas()
    for col in ("story_text", "overall_score", "language", "model_name"):
        if col not in df.columns:
            sys.exit(f"[slop] FATAL: expected column '{col}' missing. cols={list(df.columns)}")
    n0 = len(df)
    df = df[df["language"] == "en"].copy()
    df = df.drop_duplicates(subset="story_text")
    df = df.dropna(subset=["overall_score", "story_text"])
    print(f"[slop] rows {n0} -> en+dedup {len(df)}", flush=True)
    cands = []
    for model, g in df.groupby("model_name"):
        q25 = g["overall_score"].quantile(0.25)
        tail = g[g["overall_score"] <= q25]
        tail = tail.sort_values("overall_score")            # worst first
        taken = 0
        for _, row in tail.iterrows():
            if taken >= 25:
                break
            cs = chunk_text(str(row["story_text"]))
            if cs:
                cands.append(dict(text=cs[0], source_id=f"model:{model}", wc=wc(cs[0]), era="modern_ai"))
                taken += 1
        if taken:
            print(f"[slop] model:{model} q25={q25:.3f} -> {taken} pooled", flush=True)
    return cands


# ---------------------------------------------------------------- EUCLAISE band
def build_euclaise(n=60):
    from datasets import load_dataset
    from sacremoses import MosesDetokenizer
    md = MosesDetokenizer(lang="en")
    print("[band] loading euclaise/writingprompts ...", flush=True)
    ds = load_dataset("euclaise/writingprompts", split="train")
    field = "story" if "story" in ds.column_names else ds.column_names[-1]
    cands = []
    for i, row in enumerate(ds):
        if len(cands) >= n * 3:
            break
        txt = row[field]
        if not txt:
            continue
        txt = md.detokenize(str(txt).split())               # fix 'wo n't', ' . '
        cs = chunk_text(txt)
        if cs:
            cands.append(dict(text=cs[0], source_id="euclaise", wc=wc(cs[0]), era="modern_human"))
    return cands


# ---------------------------------------------------------------- sampling
def stratified_sample(cands, n=N_PER_CLASS, cap=PER_SOURCE_CAP, nbins=N_BINS):
    if not cands:
        return [], {}, {}
    edges = np.linspace(LO, HI, nbins + 1)
    for c in cands:
        c["bin"] = int(min(nbins - 1, max(0, np.searchsorted(edges, c["wc"], side="right") - 1)))
    order = list(range(len(cands)))
    rng.shuffle(order)
    cands = [cands[i] for i in order]
    per_bin = n // nbins
    sel, src_ct, bin_ct = [], {}, {b: 0 for b in range(nbins)}
    # pass 1: even bins under caps
    for c in cands:
        if len(sel) >= n:
            break
        if bin_ct[c["bin"]] >= per_bin:
            continue
        if src_ct.get(c["source_id"], 0) >= cap:
            continue
        sel.append(c); src_ct[c["source_id"]] = src_ct.get(c["source_id"], 0) + 1; bin_ct[c["bin"]] += 1
    # pass 2: fill deficit from any bin, still under caps
    if len(sel) < n:
        chosen = {id(c) for c in sel}
        for c in cands:
            if len(sel) >= n:
                break
            if id(c) in chosen:
                continue
            if src_ct.get(c["source_id"], 0) >= cap:
                continue
            sel.append(c); src_ct[c["source_id"]] = src_ct.get(c["source_id"], 0) + 1; bin_ct[c["bin"]] += 1
    return sel, src_ct, bin_ct


def write_pile(label, sel, manifest_rows, band="main"):
    d = os.path.join(REPO, "data", "passages", label)
    os.makedirs(d, exist_ok=True)
    for i, c in enumerate(sel):
        pid = f"{label}_{i:03d}"
        with open(os.path.join(d, pid + ".txt"), "w") as f:
            f.write(c["text"])
        manifest_rows.append(dict(
            passage_id=pid, **{"class": label}, source_id=c["source_id"],
            word_count=c["wc"], char_len=len(c["text"]), era=c["era"],
            chunk_method="sent_budget", band=band))


def report(label, sel, src_ct, bin_ct):
    if not sel:
        print(f"[{label}] !!! 0 passages"); return
    wcs = np.array([c["wc"] for c in sel])
    print(f"[{label}] N={len(sel)} wc mean={wcs.mean():.0f} sd={wcs.std():.0f} "
          f"med={np.median(wcs):.0f} | bins={dict(bin_ct)}")
    top = sorted(src_ct.items(), key=lambda x: -x[1])[:6]
    print(f"[{label}] sources={len(src_ct)} top={top} maxshare={max(src_ct.values())/len(sel):.0%}")


def add_piles(names):
    """Phase 1: APPEND new pile(s) to the existing corpus without touching frozen great/flat/slop."""
    import pandas as pd
    mpath = os.path.join(REPO, "data", "manifest.csv")
    man = pd.read_csv(mpath)
    rows = []
    for name in names:
        if name == "modhuman":
            cands = build_modhuman()
            sel, src, binc = stratified_sample(cands, n=80, cap=PER_SOURCE_CAP)
        elif name == "modgreat":
            cands = build_great(MODERN_GREAT_IDS, "1922_1930", "modgreat")
            sel, src, binc = stratified_sample(cands, n=80, cap=PER_SOURCE_CAP)
        else:
            print(f"[add] unknown pile '{name}', skipping"); continue
        if name in set(man["class"]):
            print(f"[add] pile '{name}' already in manifest, skipping (delete it first to rebuild)"); continue
        report(name, sel, src, binc)
        write_pile(name, sel, rows, band="main")
    if not rows:
        print("[add] nothing added"); return
    new = pd.DataFrame(rows)
    out = pd.concat([man, new], ignore_index=True)
    out.to_csv(mpath, index=False)
    print(f"=== appended {len(rows)} passages; manifest now {len(out)} ===", flush=True)
    print("[length check] mean word_count by class (main band):")
    print(out[out.band == "main"].groupby("class")["word_count"].agg(["count", "mean", "std"]).round(1).to_string())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", default=None, help="comma list of new piles to append: modhuman,modgreat")
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
    if args.add:
        return add_piles([x.strip() for x in args.add.split(",") if x.strip()])
    print("=== building candidate pools ===", flush=True)
    great = build_great()
    flat = build_flat()
    slop = build_slop()
    band = build_euclaise()
    print("=== stratified length-matched sampling (N=80/class, cap 15%, 5 bins) ===", flush=True)
    rows = []
    for label, cands in [("great", great), ("flat", flat), ("slop", slop)]:
        sel, src_ct, bin_ct = stratified_sample(cands)
        report(label, sel, src_ct, bin_ct)
        write_pile(label, sel, rows, band="main")
    bsel, bsrc, bbin = stratified_sample(band, n=60, cap=60)
    report("euclaise", bsel, bsrc, bbin)
    write_pile("euclaise", bsel, rows, band="era_band")

    mpath = os.path.join(REPO, "data", "manifest.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["passage_id", "class", "source_id", "word_count",
                                          "char_len", "era", "chunk_method", "band"])
        w.writeheader(); w.writerows(rows)
    print(f"=== wrote {len(rows)} passages to manifest.csv ===", flush=True)
    # cross-class length sanity (the confound the residualizer must also defend against)
    import pandas as pd
    m = pd.read_csv(mpath)
    print("[length check] mean word_count by class:")
    print(m[m.band == "main"].groupby("class")["word_count"].agg(["count", "mean", "std"]).round(1).to_string())


if __name__ == "__main__":
    main()
