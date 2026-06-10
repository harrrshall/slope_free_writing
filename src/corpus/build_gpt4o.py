#!/usr/bin/env python3
"""Build a GPT-4o AI-fiction pile (Gryphe/ChatGPT-4o-Writing-Prompts) as a 2nd AI source (Z2 Part B).
Genre-matched to modhuman (WritingPrompts fiction), DIFFERENT generator (GPT-4o, NOT in lars1234's 15).

HONEST CAVEAT: this is a SINGLE model -> grouped by STORY (source_id = story), so it tests whether F5 holds
for one FRONTIER generator, NOT cross-model generality (it does not, by itself, close the 15-model ceiling).
Length-matched (same randomized [150,400] chunker as the frozen corpus) + typography-normalized at source (M13).
"""
import os
import re
import numpy as np
import pandas as pd
import spacy

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ root on path for library imports
from typography import normalize_typography

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LO, HI = 150, 400
rng = np.random.default_rng(0)
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
nlp.max_length = 2_000_000


def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()[:120_000]
    if not text:
        return []
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    chunks, buf, n = [], [], 0
    target = int(rng.integers(LO, HI + 1))
    for s in sents:
        sw = len(s.split())
        if not buf and sw > HI:
            continue
        buf.append(s); n += sw
        if n >= target or n >= HI:
            if LO <= n <= HI:
                chunks.append(" ".join(buf))
            buf, n, target = [], 0, int(rng.integers(LO, HI + 1))
    if LO <= n <= HI:
        chunks.append(" ".join(buf))
    return chunks


def main(n_target=80):
    from datasets import load_dataset
    ds = load_dataset("Gryphe/ChatGPT-4o-Writing-Prompts", split="train")
    print(f"[gpt4o] cols={ds.column_names} (conversations format: the 'gpt' turn is the AI story)", flush=True)

    def story_of(row):
        """The GPT-4o story = the assistant ('gpt') turn's value in the conversation."""
        conv = row.get("conversations") or []
        vals = [t.get("value", "") for t in conv if isinstance(t, dict) and t.get("from") in ("gpt", "assistant")]
        return vals[-1] if vals else None

    rows, i_story = [], 0
    for row in ds:
        if len(rows) >= n_target:
            break
        txt = story_of(row)
        if not txt or not isinstance(txt, str):
            continue
        cs = chunk_text(normalize_typography(txt))
        if cs:
            rows.append(dict(text=cs[0], source_id=f"gpt4o_story_{i_story}", wc=len(cs[0].split())))
            i_story += 1

    d = os.path.join(REPO, "data", "passages_gpt4o", "gpt4o")
    os.makedirs(d, exist_ok=True)
    mrows = []
    for k, c in enumerate(rows):
        pid = f"gpt4o_{k:03d}"
        open(os.path.join(d, pid + ".txt"), "w").write(c["text"])
        mrows.append(dict(passage_id=pid, **{"class": "gpt4o"}, source_id=c["source_id"], word_count=c["wc"],
                          char_len=len(c["text"]), era="modern_ai", chunk_method="sent_budget", band="main"))
    pd.DataFrame(mrows).to_csv(os.path.join(REPO, "data", "manifest_gpt4o.csv"), index=False)
    print(f"[gpt4o] wrote {len(mrows)} passages, mean wc={np.mean([r['word_count'] for r in mrows]):.0f} "
          f"(modhuman ~274, slop ~277 for length-match check)", flush=True)


if __name__ == "__main__":
    main()
