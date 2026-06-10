#!/usr/bin/env python3
"""Track Z analysis: do our text-prosodic PRIMITIVES predict real reading behaviour (ZuCo eye-tracking
+ theta-EEG) BEYOND length / frequency / GPT-2 surprisal? Pre-registered EXPERIMENT_LOG 2026-06-08.

Word-level. PROSODY (ours): nsyl (CMUdict syllables), is_content (carries lexical stress / not a function
word), is_boundary (our clause/sentence boundary). CONFOUNDS: length, log-freq (wordfreq), GPT-2 surprisal,
position, sentence-final. Binding = improvement of the PROSODY block over confounds-only (LR test) + the
is_boundary coefficient. Skipped words (nFix=0) modelled separately from reading times.

`python src/zuco/zuco_analysis.py --test-surprisal`  -> sanity-check surprisal on one sentence
`python src/zuco/zuco_analysis.py`                   -> full analysis -> results/report_zuco.json
"""
import glob
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import pronouncing
import wordfreq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PUNCT_BOUNDARY = {",", ";", ":", "—", "–", "-", "(", ")", ".", "!", "?"}
STOP = set(open(os.path.join(REPO, "data", "slop_lists", "nltk_stopwords.txt")).read().split())

_tok = _model = None


def _load_gpt2():
    global _tok, _model
    if _model is None:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _tok = GPT2TokenizerFast.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return _tok, _model


def sentence_surprisals(words):
    """Per-word GPT-2 surprisal (sum of subword surprisals) for an ordered ZuCo word list.
    BOS is prepended so even the first real token has context. Returns list aligned to `words`."""
    import torch
    tok, model = _load_gpt2()
    text = " ".join(words)
    enc = tok(text, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    bos = tok.bos_token_id
    inp = torch.tensor([[bos] + ids])
    with torch.no_grad():
        logp = torch.log_softmax(model(inp).logits[0], dim=-1)   # (1+T, V)
    # surprisal of real token j (0-based in ids) is predicted from position j in `inp` (the bos-shifted seq)
    surp = [-float(logp[j, ids[j]]) for j in range(len(ids))]
    # char spans of each word in text
    spans, pos = [], 0
    for w in words:
        s = text.index(w, pos); spans.append((s, s + len(w))); pos = s + len(w)
    wsurp = [0.0] * len(words)
    for j, (a, b) in enumerate(offs):
        if a == b:
            continue
        c = a                                   # GPT-2 tokens include a leading space; skip to first real char
        while c < b and text[c] == " ":
            c += 1
        if c >= b:
            continue
        for wi, (ws, we) in enumerate(spans):
            if ws <= c < we:
                wsurp[wi] += surp[j]
                break
    return wsurp


def _core(w):
    return re.sub(r"[^A-Za-z']", "", w)


def _nsyl(core):
    ph = pronouncing.phones_for_word(core.lower())
    if ph:
        return max(1, len(pronouncing.stresses(ph[0])))
    return max(1, len(re.findall(r"[aeiouy]+", core.lower()))) if core else 1


def build_features(df):
    """Attach prosody + confound predictors per word. Surprisal computed once per unique sentence."""
    rows = []
    df = df.dropna(subset=["word"]).copy()
    df["word"] = df["word"].astype(str)
    df = df[df["word"].str.strip() != ""]
    # surprisal cache keyed by sentence text (same sentences across subjects)
    surp_cache = {}
    for (subj, sidx), g in df.groupby(["subject", "sent_idx"], sort=False):
        g = g.sort_values("word_idx")
        words = g["word"].tolist()
        key = " ".join(words)
        if key not in surp_cache:
            try:
                surp_cache[key] = sentence_surprisals(words)
            except Exception:
                surp_cache[key] = [np.nan] * len(words)
        surps = surp_cache[key]
        n = len(words)
        for i, (_, r) in enumerate(g.iterrows()):
            w = r["word"]; core = _core(w)
            if not core:
                continue
            trailing = w[len(w.rstrip(".,;:!?—–-()")):] if w else ""
            is_boundary = int(bool(set(w[-1:]) & _PUNCT_BOUNDARY) or i == n - 1)
            rows.append(dict(
                subject=subj, sent_idx=sidx, word_idx=r["word_idx"], word=w,
                FFD=r["FFD"], GD=r["GD"], GPT=r["GPT"], TRT=r["TRT"], nFix=r["nFix"], theta=r["theta_trt"],
                # prosody (ours)
                nsyl=_nsyl(core), is_content=int(core.lower() not in STOP), is_boundary=is_boundary,
                # confounds
                wlen=len(core), logfreq=wordfreq.zipf_frequency(core.lower(), "en"),
                surprisal=surps[i] if i < len(surps) else np.nan,
                word_pos=i / max(1, n - 1), is_sent_final=int(i == n - 1)))
    return pd.DataFrame(rows)


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


CONF = ["wlen", "logfreq", "surprisal", "word_pos", "is_sent_final"]
PROS = ["nsyl", "is_content", "is_boundary"]


def _fit_block(y, X_conf, X_pros, clusters, logit=False):
    """Confounds(+subject FE) vs +prosody, with SENTENCE-clustered robust SE. Returns the full model and
    a Wald F-test on the joint prosody block (valid under robust SE, unlike a likelihood-ratio test)."""
    import statsmodels.api as sm
    Xc = sm.add_constant(X_conf, has_constant="add")
    Xf = sm.add_constant(pd.concat([X_conf, X_pros], axis=1), has_constant="add")
    ck = {"cov_type": "cluster", "cov_kwds": {"groups": clusters}}
    if logit:
        mf = sm.Logit(y, Xf).fit(disp=0, **ck)
    else:
        mf = sm.OLS(y, Xf).fit(**ck)
    restr = ", ".join(f"{t} = 0" for t in X_pros.columns)
    w = mf.wald_test(restr, scalar=True)
    return mf, float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))


def run_models(d):
    out = {}
    d = d.copy()
    for c in CONF + PROS:
        d[c + "_z"] = _z(d[c])
    conf_cols = [c + "_z" for c in CONF]
    pros_cols = [c + "_z" for c in PROS]
    subj_fe = pd.get_dummies(d["subject"], prefix="S", drop_first=True).astype(float)
    d["fixated"] = (pd.to_numeric(d["nFix"], errors="coerce") > 0).astype(int)

    def block(name, y, mask, logit=False):
        Xc = pd.concat([d.loc[mask, conf_cols], subj_fe.loc[mask]], axis=1)
        Xp = d.loc[mask, pros_cols]
        try:
            mf, F, p = _fit_block(y[mask], Xc, Xp, d.loc[mask, "sent_idx"], logit=logit)
        except Exception as e:
            out[name] = {"error": str(e)[:120]}; return
        out[name] = {"prosody_block_F": F, "prosody_block_p": p, "n": int(mask.sum()),
                     "is_boundary_coef": float(mf.params.get("is_boundary_z", np.nan)),
                     "is_boundary_p": float(mf.pvalues.get("is_boundary_z", np.nan)),
                     "prosody_coefs": {c: round(float(mf.params.get(c + "_z", np.nan)), 4) for c in PROS},
                     "prosody_p": {c: round(float(mf.pvalues.get(c + "_z", np.nan)), 4) for c in PROS}}

    # (1) SKIP: did the word get fixated at all
    block("skip", d["fixated"].astype(float), d["surprisal"].notna(), logit=True)
    # (2) READING TIME: log total reading time, fixated words only
    f_mask = (d["fixated"] == 1) & d["surprisal"].notna() & (pd.to_numeric(d["TRT"], errors="coerce") > 0)
    d["logTRT"] = np.log(pd.to_numeric(d["TRT"], errors="coerce").clip(lower=1))
    block("reading_time", d["logTRT"], f_mask)
    # (3) THETA EEG (fixated words)
    d["theta_z"] = _z(d["theta"])
    block("theta_eeg", d["theta_z"], f_mask & d["theta"].notna())
    return out


def main():
    files = sorted(glob.glob(os.path.join(REPO, "data", "zuco", "words_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"[zuco] {len(df)} words, {df['subject'].nunique()} subjects, {df['sent_idx'].nunique()} sentences")
    feats = build_features(df)
    print(f"[zuco] features built: {len(feats)} words; surprisal non-null {feats['surprisal'].notna().mean():.2f}")
    # persist the analysis-ready per-word table (prosody + confounds + surprisal + ET + EEG) so later
    # analysis never has to recompute GPT-2 surprisal. Also a combined raw-words table for convenience.
    feats.to_parquet(os.path.join(REPO, "data", "zuco", "features_zuco.parquet"), index=False)
    df.to_parquet(os.path.join(REPO, "data", "zuco", "words_all.parquet"), index=False)
    print(f"[zuco] saved data/zuco/features_zuco.parquet ({feats.shape}) + words_all.parquet ({df.shape})")
    res = run_models(feats.dropna(subset=["surprisal"]).reset_index(drop=True))
    res["n_words"] = int(len(feats)); res["n_subjects"] = int(df["subject"].nunique())
    res["n_sentences"] = int(df["sent_idx"].nunique())

    def verdict(r):
        rt = r.get("reading_time", {})
        real = (rt.get("prosody_block_p", 1) < 0.01) or \
               (rt.get("is_boundary_p", 1) < 0.01 and rt.get("is_boundary_coef", 0) > 0)
        return "PROSODY-REAL" if real else "NULL (prosody adds nothing beyond length/freq/surprisal)"
    res["verdict"] = verdict(res)
    with open(os.path.join(REPO, "results", "reports", "report_zuco.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    lines = [f"\n## 2026-06-08 — Track Z RESULT (ZuCo implicit-prosody validation): VERDICT {res['verdict']}",
             f"- {res['n_words']} words, {res['n_subjects']} subjects, {res['n_sentences']} Wikipedia sentences (NR); "
             f"prosody=ours(nsyl,is_content,is_boundary) vs confounds(length,logfreq,GPT2-surprisal,pos,sent-final)+subjectFE, sentence-clustered SE"]
    for k in ("skip", "reading_time", "theta_eeg"):
        v = res.get(k, {})
        if "prosody_block_p" in v:
            lines.append(f"- {k}: prosody-block F={v['prosody_block_F']:.1f} p={v['prosody_block_p']:.2e} | "
                         f"is_boundary coef={v.get('is_boundary_coef', float('nan')):+.3f} p={v.get('is_boundary_p', float('nan')):.2e} | "
                         f"coefs={v.get('prosody_coefs')} (n={v['n']})")
        elif v:
            lines.append(f"- {k}: {v.get('error','(no fit)')}")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(REPO, "docs", "EXPERIMENT_LOG.md"), "a") as fh:
        fh.write(summary)
    print(summary)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-surprisal":
        s = ["The", "cat", "sat", "on", "the", "supercalifragilistic", "mat."]
        print(list(zip(s, [round(x, 2) for x in sentence_surprisals(s)])))
    else:
        main()
