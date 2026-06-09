# Extraction-Upgrade Plan — is the bottleneck the signal, or how we measure it?

> Status doc written 2026-06-08. Self-contained summary of where the project stands,
> what we found, and the next experimental arc. Technical depth intended. Read alongside
> `FINDINGS.md` (F1/F2), `DECISIONS.md` (D14), `MISTAKES.md` (M12/M13/M14), and
> `AGENTS.md` §1 (research stance).

---

## 0. The frame (do not lose this)

**North Star:** a better reward model for writing quality, usable in RLHF / preference
optimization to produce slop-free writing *without collapsing diversity*. Writing has no
verifiable reward; LLM-as-judge is biased toward slop (M2). We hunt for quality signals that
are real and hard to game.

**Research stance (AGENTS.md §1):** open-ended discovery, not application of a predefined
solution. Prosody is the current *probe*, not the destination. Adopting any standard approach
must be earned by evidence. We are explicitly willing to conclude "prosody is weak" if that is
what the data says, and to follow an unplanned result wherever it points.

**This document's question:** Phase 1 left the within-human quality signal capped at ~0.65.
That number has two incompatible explanations, and we cannot yet tell them apart:
1. the signal genuinely is weak (prosody does not carry much within-human quality information), or
2. our **extraction is too crude to see it** (we are measuring prosody with a blunt instrument).

The plan below is designed to discriminate between (1) and (2) cleanly.

---

## 1. What we did, and what it showed

### Phase 0 — does any prosody signal exist? (H1) → PROCEED

- **Method:** Tier-A text-prosody features (stress timing, syllable distribution, clause
  spacing, sentence-length variation) → `StandardScaler` + logistic regression, group-aware
  5-fold CV (group = source book / Reuters category / model), balanced-accuracy, group-level
  permutation null (n=1000), quarantined held-out scored once.
- **Result (F1):**
  - great-vs-flat balanced accuracy **0.984** (held-out 0.969), group-perm **p=0.001**; 3-class 0.917.
  - length-only and MATTR-only baselines at **chance** (0.511 / 0.536); prosody beats best baseline
    by **+0.448**; fold-internal residualization (regress out sentence-length-mean + MATTR per train
    fold) still **0.961**.
  - within-narrative great-vs-slop **0.930** (held-out 0.908), and **0.874 on sub-sentence rhythm
    features only** (no sentence-length, no vocab/OOV).
  - direction correct: great prose ~5.5x the sentence-length variance of slop (243 vs 44), higher
    stress-run-length variance → rhythmic VARIATION, not metrical regularity (M3 guard passed).
- **Limit:** great-vs-flat (0.984) is genre/era confounded (pre-1928 fiction vs 1987 finance wire).
  The cleaner quality contrast is great-vs-slop (0.91-0.93).

### Phase 1 + 1c — is it orthogonal to lexical slop, and confound-controlled? (H2/H3) → PROCEED, hardened

- **Method:** introduced a competent-modern-human pile (`modhuman` = frozen Fan-2018 r/WritingPrompts,
  human by construction, D10) and an exploratory acknowledged-great-modern pile (`modgreat`,
  1922-1930 Gutenberg, D13). Defined a clean SLOP-ONLY lexical baseline (slop-word / n-gram densities
  + readability incl. word length, D12). Binding test = prosody's *residual / increment over the
  lexical baseline* on fair contrasts where lexical cannot trivially cheat.
- **The key correction (M12):** the original "prosody vs FULL-lexical" KILL on great-vs-slop (0.56)
  was an artifact of putting diversity proxies (distinct-n, MATTR) into the lexical baseline and of
  testing on a near-ceiling/circular contrast (the slop lists were built from AI text). Against a
  genuine slop-only baseline on fair contrasts, the signal survives.
- **Result (F2), after the Phase-1c hardening pass** (typography normalized at source across all
  piles, flat de-duplicated, perm n=1000; independently re-derived by 6 adversarial verifier agents):
  - binding **great-vs-modhuman** residual-vs-slop **0.653** (CI [0.571, 0.732], p=0.001), held-out
    **0.695**; increment of prosody over the slop baseline **+0.102** (CI [0.034, 0.168], significant).
  - **modhuman-vs-slop** (modern human vs AI) prosody-only **0.891** dev / **0.904** held-out (p=0.001).
  - typography control: a content-free typography-only classifier drops 0.758 → 0.481 on
    great-vs-modhuman after normalization, while prosody features are unchanged (M13 resolved at source).
- **What F2 means:** prosody is *partly* independent of lexical-slop detection. The strongest, most
  robust signal is rhythmic **variation** distinguishing modern-human from AI prose. The within-human
  *quality* gradient (great-vs-modhuman) is real but weak (~0.65).
- **Limit:** this is human-vs-AI separation plus a weak within-human gradient; it is NOT a demonstration
  that prosody ranks fine-grained writing quality. great-vs-flat (0.98) is inherently genre-confounded.

### The decision that triggered this doc — D14

Prosody is **NOT** a standalone quality reward: its unique within-human contribution is small and it is
trivially gameable (rewarding sentence-length variance invites erratic-but-bad text). It is at most ONE
term in a composite, multi-signal reward, with a diversity-preserving optimizer (D6) as the structural
defense. **D14's open question, which this plan answers:** is the ceiling the *model / representation*,
not prosody itself? Verification already showed the classifier *head* is not the bottleneck (RF/SVM/seed
swaps matched logreg). The untested lever is the **representation / extraction**.

---

## 2. The current extraction, and why it is a crude proxy

All features today come from `features_lib.py`. Technically, what we compute is a **first-order lexical
proxy** for prosody, then collapse it to 14 passage-level summary statistics (`PROSODY_COLS`):

| Group | Features | How computed | Weakness |
|---|---|---|---|
| stress_timing | gap_mean/var/cv, stress_runlen_var, stress_bin_entropy | `pronouncing.phones_for_word(w)[0]` → binary stress over syllables, diffed across the **whole passage** | **Citation stress**: dictionary stress of each word in isolation. Ignores phrasal/nuclear prominence, function-word destressing, the rhythm rule (THIR-teen → THIRteen MEN), noun/verb shifts (REcord/reCORD). Takes only the first pronunciation. **No phrase reset** — the passage is one long stress string. |
| sentence_rhythm | sl_var, sl_cv | spaCy sentence segmentation, count alpha tokens | OK but coarse; order discarded. |
| clause_spacing | ph_mean, ph_var | gaps between **punctuation** boundaries (`, ; : - ( )` + sent-end) | Punctuation is a crude stand-in for *prosodic* phrasing; real prosodic breaks do not map 1:1 to commas. |
| syllable_dist | spw_mean/var/skew | syllable count per word from CMUdict | reasonable; still summary-only. |
| confound | sl_mean, oov_rate | — | OOV words are **dropped** from the stress sequence → era/register bias. |

Two structural problems sit on top of the per-feature ones:

1. **Citation stress ≠ prosodic prominence.** The entire TTS/prosody field moved past dictionary stress
   more than a decade ago. Prominence in connected language is contextual, syntactic, and
   information-structural; the dictionary cannot see any of it.
2. **Summary stats discard sequence.** "sentence-length variance = 243" says there is variation but not
   the *pattern*: long-short-long-short and long-long-long-short have identical variance and read nothing
   alike. Prose music is a sequence; we are measuring it with histograms.

This is exactly the kind of measurement that could produce a false ~0.65 ceiling.

---

## 3. The upgrade path (web-verified, all text-only, honors M6/D4 — no audio)

### Component A — speech-grounded per-word prominence (the headline change)

- **Helsinki-NLP/prosody** (Talman et al. 2019, NoDaLiDa; arXiv:1908.02262; MIT code, CC-BY-4.0 corpus).
  BERT predicts, **from raw text**, a per-word prominence level (0 non-prominent / 1 prominent /
  2 highly prominent), a **real-valued** prominence score, and a per-word **boundary** label.
- **Why it is better:** the labels are not hand-rules. They were derived from *real human speech*
  (LibriTTS audiobooks) via Continuous Wavelet Transform annotation. So we get a prominence-per-word
  sequence grounded in how humans actually realize rhythm aloud, replacing dictionary citation stress.
- **Usability:** no pretrained checkpoint ships, but the training code and corpus do. Train once on
  their corpus, **freeze** it, and use it purely as a feature extractor. Reported accuracy 83.2% (2-way)
  / 68.6% (3-way). Deps: PyTorch, pytorch_transformers, numpy.
- **Data-scale safety:** the extractor is trained on the *large Helsinki corpus*, never on our ~80/class
  contrast set. Our small data only ever sees the frozen extractor's outputs.

### Component B — multiscale rhythm via Continuous Wavelet Transform (the real "prose music")

- **Suni/Vainio CWT work + `asuni/wavelet_prosody_toolkit`.** Prosody lives at multiple timescales
  simultaneously (word → phrase → sentence → paragraph); CWT decomposes a 1-D signal into a scalogram
  across scales. The toolkit defaults to audio, but CWT applies to any 1-D series.
- **Use:** run CWT on the text-derived prominence/boundary sequence from Component A, read off multiscale
  rhythm features. This is the principled version of `gap_var` / `stress_runlen_var`: it captures the
  *timescale and pattern* of rhythmic variation that summary stats flatten. Biggest conceptual upgrade.

### Component C — learned prosodic phrase breaks (replace the punctuation proxy)

- The Helsinki corpus already carries boundary labels; the same model yields predicted prosodic breaks.
  Strong separate literature (BERT phrase-break prediction: Japanese 94.3% break-index accuracy;
  multilingual transfer). Replaces `ph_*` punctuation gaps with predicted prosodic phrasing.

### Adjacent, lower priority
- `ProseRhythmDetector` (rhetorical repetition figures — a *different* sense of rhythm; possible later axis).
- arXiv:2411.04950 "sequentially correlated literary properties" — supports the sequence-over-summary direction.

---

## 4. On model choice — "BERT is old, should we use a newer / our own model?"

- **BERT is a FROZEN extractor here, not the product.** We keep the prominence sequence and discard the
  model. The binding constraints are (1) whether prosodic prominence carries within-human quality signal
  at all, and (2) the label definition (3-level CWT-derived prominence). The ceiling is the labels and the
  science, **not** the encoder generation. A newer encoder (ModernBERT/DeBERTa-v3/LLM) on the same 3-level
  task buys marginal accuracy where the uncertainty does not live → cheap ablation *after* Stage 1.
- **The legitimate "modern" target is richer outputs, not a newer encoder.** A modern neural TTS front-end
  (FastSpeech2 / StyleTTS2-style) predicts *continuous, multi-dimensional* prosody from text — per-unit
  duration, pitch (F0) contour, energy — with **no audio rendered** (M6/D4-clean). Strictly richer than
  3-level prominence. Slot as a Stage 2.5/3 ablation against the frozen-BERT extractor.
- **Counterintuitive caution (important):** for *this* goal, bigger/newer can be **worse**. A larger,
  LLM-based extractor reads more lexical/semantic context to predict prosody, so its output is *more
  entangled* with lexical content — which is exactly the thing we need prosody to be orthogonal to (H2 /
  M12). A higher raw prominence-accuracy number from a bigger model can therefore *lower* the quality of
  the scientific answer. Prefer a smaller, prosody-specialized extractor; judge by **increment over
  lexical**, never raw accuracy.
- **Never train an extractor from scratch on our data.** ~80/class would relearn confounds. Any "train our
  own" path = fine-tune a pretrained model on the *large* Helsinki corpus, never on the contrast set.

---

## 5. The plan — staged, cheapest-kill-first (M9), gauntlet intact

Every stage is judged on the clean within-human contrast **great-vs-modhuman**, through the existing
Phase-1c gauntlet (typography normalized at source, group-aware CV, group-perm null n=1000, quarantined
held-out scored once), with **lexical-only and typography-only baselines sitting alongside**. The binding
number is prosody's **increment over the lexical baseline**, pre-registered, held-out confirmed — never
raw accuracy.

| Stage | Cost | What | Pass → | Fail →|
|---|---|---|---|---|
| **1** | ~1 day | Drop-in extractor swap: frozen Helsinki prominence replaces dictionary stress; recompute the **same** 14 summary stats; re-run existing harness. | speech-grounded summary stats beat dictionary-stress summary stats → extraction *was* part of the problem; go to Stage 2 | no lift → already informative; the crude-extraction story weakens |
| **2** | few days | Add CWT multiscale rhythm features (Component B) on the prominence + boundary sequences. | sequence/timescale structure adds increment-over-lexical → build Stage 3 | no lift → summary stats already captured what order carries |
| **2.5** | ablation | Swap the extractor for a modern TTS front-end (continuous duration/pitch/energy) and/or a newer encoder; compare against frozen BERT. | richer targets raise increment | no lift → encoder/targets were not the lever |
| **3** | week+ | Sequence model (HMM/n-gram → 1D-CNN/GRU → tiny Transformer, in that data-efficient order) on the *better* sequence. | learned sequence rep beats hand features on the clean contrast | no lift → close the representation question |

### STATUS UPDATE (2026-06-08): the ORDER axis was tested first (cheapest-kill) and KILLED

Before the Helsinki value-axis swap, we ran the cheaper M9 precursor on the **order axis**: 10 pre-registered
order-aware hand features (autocorrelation / trend / multiscale / turning-point / runs-z) on the *existing*
SL/SPW/SYMS/PHG series. Result = **clean KILL, verified robust by 4 adversarial agents** (FINDINGS F3,
DECISIONS D16, EXPERIMENT_LOG 2026-06-08): all three gates fail, and order residualized against the summary
histograms collapses to chance (0.551, below the residualized shuffle 0.567) → order is fully redundant with
the histograms; no model/seed/subset rescue clears it.

**Consequence for this plan:** Stage 2's *CWT/sequence* component and Stage 3 (sequence model) are
**de-prioritized for these series** — a richer ORDER encoder on the same dictionary-stress / sentence-length
series will not clear the group-level noise (D16, M9). What remains genuinely open is the **VALUES axis**,
not order: the Helsinki *speech-grounded per-unit prominence* (Stage 1/2's original headline — does a better
per-unit signal, independent of its ordering, lift the contrast?), the **learned-representation ceiling
probe** (D14), and the **higher-ceiling within-genre/era dataset** (§6), now doubly motivated by the power
limit this run surfaced (n=127 / 70 groups → CIs ~±0.10).

**The decisive read, either way:**
- If a speech-grounded, multiscale, learned-sequence extractor **still** cannot beat the lexical baseline
  on great-vs-modhuman → **strong, publishable negative**: the within-human prosodic quality signal is
  genuinely weak, not merely mis-measured. D14's open question closes honestly.
- If it **does** lift the clean contrast meaningfully → **extraction was the bottleneck**; prosody carries
  more within-human quality information than F2 credited, and the next question becomes *what* the better
  representation is using (interpretability), plus whether it stays orthogonal to lexical.

---

## 6. The deeper limiter (do not forget while chasing extraction)

Even the best extractor is judged on the contrasts we have, and our cleanest within-human contrast is
still **amateur-Reddit (`modhuman`) vs old-classics (`great`)** — confounded in register and era. The
strongest version of this whole program needs a **within-genre, within-era human quality gradient**
(e.g. published-literary vs slush-pile/amateur of the same era and genre, or professionally-edited vs
unedited prose). That single dataset would sharpen the answer for the *existing* hand-feature pipeline
*and* is the data any learned model would ultimately need. Extraction is the cheaper lever to test first
(Stage 1 is a day); the dataset is the higher-ceiling investment after Stage 1 reports.

---

## 7. Open decisions / next action

- **DONE (2026-06-08):** repo restructure logged (D15); order-axis cheapest-kill pre-registered, run, and
  adversarially verified → **KILL-robust** (F3, D16, M15). Artifacts: `src/sequence_features.py`,
  `extract_sequence.py`, `stage1_sequence.py`, `tests/test_sequence.py`, `results/features_seq_A.parquet`,
  `results/report_stage1.json`, `results/report_stage1_verification.json`. Held-out NOT touched (gates failed).
- **The order axis is closed.** Do not build CWT/sequence models on these hand series (D16/M9).
- **DIRECTION DECIDED 2026-06-08 (D17) — supersedes the three-lever menu below:** the binding problem is the
  contrast's CONFOUND (Reddit-vs-classics = register+era), which poisons positives, not just its power. So:
  (1) **de-confound the data via a WITHIN-AUTHOR quality contrast** (same author's strong vs neglected works) —
  many authors paired (group=author) to keep power, external noisy quality label (anthology/syllabus/in-print
  vs neglected), canonical-vs-neglected of the SAME subgenre (not "commercial"/"early"); (2) then **exactly ONE
  extraction test, the VALUES axis** (Helsinki speech-grounded per-unit prominence replacing dictionary stress)
  on that clean contrast — NO sequence/representation models; (3) **pre-commit a power target, the ≥20-seed
  shuffle null (M15), and a STOP RULE** before collecting; (4) **promote the human-vs-AI robustness pivot**
  (prosody as a paraphrase-robust origin/style signal — the strong 0.89 axis, novel where lexical detectors
  collapse under paraphrase) from fallback to a near-term primary, since it is cheap and runs on existing data.
  Open design questions (power-vs-confound reconciliation, label source, pivot-vs-quality sequencing) tracked in
  D17. The Helsinki VALUES test and the learned-rep probe (D14) survive as below but are now gated on the
  de-confounded dataset; the order/sequence stages are dropped (D16).
- **Standing rules that bind this work:** reward variation not regularity (M3); text-only, no audio
  (M6/D4); real human prose + quarantined held-out (M4); control length/vocab/era/genre (M5); orthogonality
  judged on fair, non-circular contrasts (M12); kill criterion before the run (M7); multi-seed nulls for
  shuffle/permutation controls (M15); web-verify every external resource (AGENTS.md §2).
