# Mistakes & Lessons

Read this **before** every task. Append to it **after** any error, dead end, or surprise.
The point is that no mistake is made twice. Each entry: what happened → why → the rule now.

The list below is **seeded** with traps identified before any code was written — from the
reasoning that motivated this project. Treat them as live warnings, not history.

---

### M1 — Optimizing a writing reward with vanilla RLHF collapses diversity
- **What/why:** Standard RLHF/DPO is mean-seeking (reverse-KL is mode-seeking) and over-represents
  majority preference; it converges onto safe high-reward modes and kills the tails where good
  writing lives. Documented across the literature.
- **Rule now:** any policy-training phase must use diversity-preserving methods (DivPO-style
  selection, forward-KL/JS) and must *measure* diversity, not just quality.

### M2 — Optimizing against an LLM-judge amplifies slop (Goodhart)
- **What/why:** LLM judges prefer overwrought, "vocab-maxxing" prose — the very slop we fight. Train
  against that judge and the score rises while writing gets worse.
- **Rule now:** never optimize against the evaluator; keep a held-out judge quarantined; include a
  judge-bias check as a diagnostic.

### M3 — Rewarding metrical regularity instead of rhythmic variation
- **What/why:** The intuitive prosody reward prefers a steady beat. That just trades one monotony
  for another; good prose has controlled *variation*.
- **Rule now:** calibrate `r_prosody` so a metronome-like passage scores low; reward variation/cadence.

### M4 — Evaluating quality on synthetic/AI text only
- **What/why:** A model scores high on text it resembles and proves nothing.
- **Rule now:** every eval set must contain real human prose; held-out and quarantined.

### M5 — Mistaking a confound for the signal
- **What/why:** "Great" prose (e.g. Gutenberg classics) differs from "flat" prose in era, sentence
  length, and vocabulary as well as rhythm. A separation result might be just "old books, long
  sentences."
- **Rule now:** control for length, vocabulary, era, genre before believing any separation (Phase 1/H3).

### M6 — Routing prosody through audio (TTS)
- **What/why:** Synthesizing speech to "hear" the prose adds no literary information (prosody is
  predicted from the same text surface) and loses signal; speech-emotion models also reward
  surface explicitness, the opposite of literary restraint.
- **Rule now:** extract prosody features *from text*; do not generate or analyze audio.

### M7 — Treating priors as findings / moving the goalpost
- **What/why:** It is tempting to report a believed-true claim as a result, or to rationalize a
  failed kill criterion after the fact.
- **Rule now:** priors live in FINDINGS.md's prior section only; define kill criteria before runs
  and honor them; a clean negative result is a success.

### M8 — Overclaiming novelty
- **What/why:** The diversity-collapse problem and its fixes (DivPO, DPH-RL, DiverseGRPO,
  LongWriter-Zero) are already published and active. Claiming the whole thing is new collapses on
  one citation.
- **Rule now:** the delta is the *prose-rhythm reward for written text* + the *synthesis*. State it
  exactly that way (see BACKGROUND.md). Re-check the literature before claiming any gap.

### M9 — Building before the cheap kill-test
- **What/why:** Elaborate pipelines built before Phase 0 waste days if the basic signal isn't there.
- **Rule now:** Phase 0 separation pre-test gates everything. Cheapest-kill-first, always.

---

### M10 — A "uniform" chunking RULE does not give uniform length DISTRIBUTIONS
- **What/why:** Phase-0 chunker packed whole sentences to a fixed [150,400] budget. Applied to
  continuous text (Gutenberg books, AI stories) it piled every chunk near the 400 cap (~380w),
  while short Reuters wire docs stayed ~273w. Same rule, but the output word-count distributions
  differed by class (great/slop ~380 vs flat ~273) — a textbook M5 length confound baked into the
  corpus before a single classifier ran. Caught at corpus QA via the per-class word_count table.
- **Rule now:** after building piles, ALWAYS print mean/std word_count per class and confirm the
  distributions match, not just that the chunking code is shared. To length-match continuous vs
  short sources, randomize the per-chunk target length (uniform in [lo,hi]) so every pile spans
  every word-count bin; then bin-stratified sampling can equalize them. Fixing corpus construction
  before any score is seen is pre-registration-clean; fixing it after is not.

### M11 — sklearn `permutation_test_score` is degenerate when the label is constant within a CV group
- **What/why:** In Phase 0 each source group is single-class (a book is all-great, a reuters
  category all-flat, a model all-slop). `permutation_test_score(..., groups=g)` shuffles labels
  WITHIN each group, so on constant-label groups it changes nothing: every "permutation" equals the
  original, perm_mean=perm_max=true=0.985, and p=1.0000. Reported next to balanced_acc=0.984 that
  contradiction is the tell — a near-perfect classifier cannot have p=1.0. It is an artifact, not a
  negative result; blindly feeding it to the kill gate would FALSE-KILL a real signal.
- **Rule now:** when label is a function of group, use a GROUP-LEVEL permutation null: assign each
  whole group a random class label (keep its samples together), recompute grouped-CV balanced
  accuracy, repeat. That is the honest "could features separate these groups this well by chance?"
  test. (Here: true=0.985 vs perm_mean≈0.51, p=0.005.) Always sanity-check that perm_mean sits near
  chance; if it equals the true score, the permutation did nothing.

### M12 — Do not make a "residual vs a near-ceiling / circular baseline" a HARD binding kill
- **What/why:** Phase-1 H2.1 made "prosody, after the lexical-slop baseline is regressed out, must
  stay above chance on great-vs-slop" a HARD kill. But on great-vs-slop the lexical baseline is
  near-ceiling (0.976) BY CONSTRUCTION (the slop lists were built from AI text, and the slop pile IS
  AI text), so there is almost no separation headroom left for ANY feature to claim a unique residual
  share. The design's own note even flagged the raw increment there as "uninformative if
  lexical_only>=0.90", yet the residual on the SAME contrast was left as a binding hard-kill. The kill
  fired (residual ba 0.56, CI incl. chance) and is a legitimate non-orthogonality result FOR THAT
  CONTRAST, but as a hard binding gate on the most lexical-favorable, circular contrast it can kill the
  whole idea before the fair contrasts (modhuman-vs-slop, great-vs-modhuman, where lexical cannot
  cheat) are ever tested.
- **Rule now:** judge orthogonality on contrasts where the competing baseline is NOT near-ceiling or
  circular. If a baseline exceeds ~0.90 by construction, treat any increment/residual there as
  DIAGNOSTIC-ONLY, never a hard binding kill. Pre-register the binding orthogonality test on a fair
  contrast (here: modern-human vs AI, where the slop lexicon does not trivially separate the classes).
  Honor the kill that fired (M7), and define the corrected binding test as an explicit NEW
  pre-registration, not a silent goalpost move.

### M13 — Typography/register glyphs are a confound that survives "cleaning" and that group-CV and lexical-residualization both miss
- **What/why:** Independent verification found modhuman (Reddit/Fan-2018) keeps `''` dialogue quotes
  (46/80) and spaced-apostrophe artifacts (21/80) that are 0/80 in great/modgreat; curly quotes 59/80
  (great) vs 16/80 (modhuman). A content-free 4-feature typography classifier separates great-vs-modhuman
  0.89, great-vs-flat 0.87, great-vs-slop 0.78. Source-grouped CV does not catch it (it is class-wide,
  not source-specific) and the lexical-slop residualizer does not remove it (typography is not in the
  feature set). prosody survived typography control for modhuman-vs-slop (0.713) but only marginally for
  great-vs-modhuman (0.606, p=0.016).
- **Rule now:** normalize typography across ALL piles in build (collapse `''`/curly/straight quotes,
  strip spaced-apostrophe artifacts, markdown) so no classifier can separate classes on punctuation; add
  a typography-only baseline as a standing diagnostic; when a cross-source-register pile is added, prove
  a punctuation-only classifier is at chance before trusting any separation involving it.

### M14 — Confirm the actual feature DRIVER and confound set from the data, not from the hypothesis
- **What/why:** We described the F1 driver as "sentence-length variation", but report_A.json's own
  permutation_importance ranks syllable_dist ABOVE sentence_rhythm. And the F1 length control used only
  sentence-length-mean (at chance), missing mean_word_len, which separates great-vs-flat at 0.93 alone.
  Two narrative claims (the driver, and "diversity proxies cause the FULL-lexical kill") were stated more
  strongly than the numbers support, even though the headline accuracies were correct.
- **Rule now:** read the importance ranking before naming a driver; include EVERY length/vocab covariate
  that independently separates the classes (word length, not just sentence length) in the residualizer;
  state mechanisms only to the precision the evidence supports, and have an adversarial pass check the
  WORDING against the numbers, not just the numbers.

_Append new lessons below as M15, M16, … with the same what/why → rule format._
