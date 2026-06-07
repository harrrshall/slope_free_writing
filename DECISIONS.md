# Decisions (ADR-lite)

Record every non-obvious choice so it isn't silently re-litigated. Each entry: the decision,
the context, the reasoning, and the status. Newest at the bottom.

```
### D# — <decision>  (status: accepted | superseded by D# | revisit)
- Context: <what prompted the choice>
- Decision: <what we chose>
- Reasoning: <why, and what we rejected>
```

---

### D1 — Pursue prosody as the first candidate reward signal  (status: accepted)
- Context: hunting for writing-quality signals that are real and hard to game; lexical slop is
  already covered by existing tools.
- Decision: start with text-derived prosody/rhythm.
- Reasoning: it is grounded (Implicit Prosody Hypothesis), plausibly orthogonal to lexical slop,
  and appears unexplored as a reward for *written* text. Rejected alternatives for *first* slot:
  pure novelty/semantic-distance signals (slot them in Phase 4).

### D2 — Separation pre-test before any pipeline  (status: accepted)
- Context: open-ended research with limited time; high risk of building on a non-signal.
- Decision: Phase 0 separation study gates all downstream work.
- Reasoning: cheapest experiment that can kill the idea; protects against sunk cost.

### D3 — Prosody is a complementary term, never the whole judge  (status: accepted)
- Context: prose rhythm effect is subtle/loose vs poetry (per cognitive-science literature).
- Decision: design the reward so prosody supplements, never dominates, the lexical/semantic reward.
- Reasoning: honest about the signal's strength; avoids trading lexical monotony for rhythmic monotony.

### D4 — Extract prosody from text, not audio  (status: accepted)
- Context: the tempting "run it through TTS and analyze the audio" route.
- Decision: text-only feature extraction (stress, syllable, phrasing).
- Reasoning: audio adds no literary information and loses signal; see MISTAKES.md M6.

### D5 — Tooling: Prosodic + pronouncing/CMUdict + spaCy  (status: accepted)
- Context: need mature, real libraries for stress/scansion/segmentation.
- Decision: `prosodic` (metrical/scansion), `pronouncing`+`cmudict` (stress patterns), `spacy`/`nltk`
  (sentence/clause segmentation); `eSpeak` for OOV words.
- Reasoning: all verified to exist and to have been applied to prose; see ENVIRONMENT.md / BACKGROUND.md.

### D6 — Diversity-preserving optimizer for the later policy stage  (status: accepted)
- Context: vanilla RLHF collapses diversity (MISTAKES.md M1).
- Decision: use DivPO-style selection and/or forward-KL/JS divergences in Phase 3.
- Reasoning: directly targets the failure mode the North Star must avoid.

### D7 — Position novelty honestly  (status: accepted)
- Context: the diversity-collapse field is crowded.
- Decision: claim novelty only for the prose-rhythm-reward-for-text + the synthesis; cite prior work.
- Reasoning: survives reviewer/professor scrutiny; see BACKGROUND.md and MISTAKES.md M8.

### D8 — Group-level permutation test for significance  (status: accepted)
- Context: Phase-0 labels are constant within each source group (book/category/model), which makes
  sklearn `permutation_test_score(groups=...)` degenerate (p=1.0 artifact; see MISTAKES.md M11).
- Decision: estimate significance by permuting class labels at the GROUP level (each whole group
  gets a random class, samples kept together), recomputing grouped-CV balanced accuracy, n=1000.
- Reasoning: it is the correct null for group-structured, group-constant-label data and cannot be
  gamed by group identity; preserves the frozen 'permutation p<0.01' gate without changing the threshold.

### D9 — Length-matched corpora via randomized chunk target  (status: accepted)
- Context: a fixed-budget chunker piled continuous text (books, AI stories) near the 400w cap while
  short Reuters docs stayed ~270w, creating a per-class length confound (MISTAKES.md M10).
- Decision: randomize each chunk's target length uniformly in [150,400] so every pile spans all
  word-count bins; then bin-stratified sampling equalizes class length distributions (achieved
  great 272 / flat 276 / slop 277, identical 16/bin).
- Reasoning: controls the M5 length confound at the sampling stage instead of leaning entirely on
  the fold-internal residualizer; fixed before any classifier score was seen (pre-registration-clean).

### D10 — Fan-2018 WritingPrompts is M4-admissible competent-modern-human prose  (status: accepted)
- Context: Phase 1 needs a MODERN human narrative pile to break the era/genre confound; ENVIRONMENT.md
  previously flagged r/WritingPrompts as "NOT verifiable-human" (AI-contamination risk).
- Decision: use the FROZEN Fan et al. 2018 FAIR WritingPrompts release (HF `euclaise/writingprompts`,
  MIT, 303,358 rows, arXiv:1805.04833) as the competent-modern-human pile (`modhuman`); admit it under M4.
- Reasoning: the archive is a frozen 2018 research release scraped from r/WritingPrompts years before
  GPT-2 (2019) and ChatGPT (2022), so it is human BY CONSTRUCTION; the contamination worry applies only
  to a FRESH live scrape, not this frozen release. It is competent-amateur, NOT acknowledged-great (state
  that limit in FINDINGS). Supersedes the ENVIRONMENT.md line-109 worry (amended in the same change).

### D11 — MATTR is a LEXICAL feature, never on both sides of the H2 increment  (status: accepted)
- Context: the H2 orthogonality test compares prosody vs a lexical baseline; MATTR is lexical diversity.
- Decision: `mattr` belongs to LEXICAL_COLS only; it stays OUT of PROSODY_COLS (already true). The
  prosody-minus-lexical increment never counts mattr on both sides.
- Reasoning: double-counting mattr would make "prosody adds over lexical" uninterpretable (M5).

### D12 — One frozen lexical baseline spec  (status: accepted)
- Context: the design drafts proposed three incompatible lexical feature sets.
- Decision: LEXICAL_COLS = 10 features (slopword/bigram/trigram density off the vendored slop-forensics
  lists with NLTK-stopword n-gram matching; not_x_but_y_rate from antislop regex line 1; distinct2;
  distinct3; rep_topk; mattr; mean_word_len; fk_grade via CMUdict). Lists loaded only from vendored disk
  (pinned slop-forensics SHA c313f042620f027d49101da3256bd306b628071a). Drop the EQ-bench 50084-phrase
  "exact replication" feature.
- Reasoning: H2 needs a competent, reproducible, offline lexical baseline, not a bit-exact Slop-Score clone.

### D13 — modgreat (1922-1930 Gutenberg) is exploratory / non-gating  (status: accepted)
- Context: no free+legal acknowledged-great CONTEMPORARY fiction exists; 1922-1930 Gutenberg (Gatsby,
  Hemingway, Faulkner, ...) is the closest acknowledged-great-modern text available.
- Decision: build `modgreat` as EXPLORATORY only. The binding era control is within-narrative
  great-vs-modhuman plus a within-Gutenberg 19C-vs-early20C null. Judge modgreat on sub-sentence-only
  features with Hemingway+Faulkner held out (they maximize sentence-length variance = the F1 driver),
  PER_SOURCE_CAP enforced.
- Reasoning: avoids "proving" prosody with the very authors famous for the F1 driver; keeps the era
  control honest (genre fully matched, era partially shifted ~100yr, contemporary-great unavailable).

### D14 — Prosody is at most ONE term in a multi-signal reward, NOT a standalone writing-quality reward  (status: accepted)
- Context: F2 / Phase-1 showed prosody's quality signal is weak on the clean within-human contrast
  (great-vs-modhuman ~0.61 after confounds), largely redundant with lexical features, and trivially
  gameable (rewarding sentence-length variance invites erratic-but-bad text). User's position: EVERY
  reward signal is gameable, so gameability alone does not disqualify a signal; the answer is to use
  several signals, not to seek one unhackable metric.
- Decision: we will NOT use prosody as the/a standalone reward for writing quality. If used at all,
  prosody is ONE component among multiple reward signals (a composite/ensemble), with the
  diversity-preserving optimizer (D6) as the structural defense against any single term being gamed. No
  single reward signal is treated as unhackable.
- Reasoning: matches the evidence (prosody's unique contribution is small and gameable) and the principle
  that robustness comes from composition + diversity preservation, not from one perfect metric. Sharpens
  D3 (complementary, never the whole judge). See FINDINGS F2.
- Open question (revisit, do not assume): is the ceiling the MODEL/representation, not prosody itself? Our
  features are 14 hand-crafted summary stats + logistic regression; the verification already showed the
  CLASSIFIER is not the bottleneck (RF/SVM/seed swaps ~ matched logreg). The untested lever is a LEARNED
  representation. Test empirically whether a learned rep beats hand-features on the confound-controlled
  within-human contrast BEFORE concluding the quality signal is absent. Caveat: a learned black-box
  prosody scorer is harder to audit for confounds and easier to reward-hack (M2) if optimized against, so
  it is better suited to MEASUREMENT than to being a reward term.
