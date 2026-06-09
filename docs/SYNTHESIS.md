# Synthesis — what this project found about text-prosody as a writing-quality signal

> The consolidated, honest story. Read this for the whole arc; `FINDINGS.md` has each finding with its
> evidence and limits, `EXPERIMENT_LOG.md` has the dated detail, `DECISIONS.md` the choices, `MISTAKES.md`
> the traps. Last updated 2026-06-08.

## One-paragraph bottom line

Text-derived prosody is **real but weak**. It robustly separates human prose from *weaker* AI and tracks
register/era, but it does **not** finely rank within-human writing quality, and its human-vs-AI edge
**degrades as models improve** (it nearly disappears for GPT-4o). We ruled out the obvious "we're processing
it wrong" explanations — the classifier (D14) and the order of the features (F3) — and a speech-grounded
prominence front-end (F7) barely moved detection while giving only a *borderline* lift to the quality
contrast. The honest conclusion: prosody is at best **one minor, complementary term** in a composite reward
(anti-monotony / rhythmic variation), not a standalone quality signal (D14). The two questions we could
**not** settle — does a *learned representation* beat hand-formulas, and does a *de-confounded, higher-power*
dataset change the within-human quality picture — both run into the same wall: **data scale and a confounded
contrast.**

## The question (North Star)

Writing has no verifiable reward, and LLM-as-judge is biased toward the bland "slop" we want to remove. The
project hunts for a writing-quality signal that is real and hard to game. The first candidate: **text-derived
prosody** — the rhythm a reader hears internally while reading silently — hypothesized to carry quality
information that lexical/semantic reward models miss. The instant goal was to test that cheaply and honestly
before building anything, and to honor a clean negative result as a success.

## Method discipline (what makes the numbers trustworthy)

Group-aware cross-validation (group = source book / model / prompt) so a classifier can't cheat on source
identity; **group-level** permutation nulls and **group bootstrap** CIs (passage-level would understate
uncertainty); **pre-registered** kill criteria frozen before each run (M7); the binding number is always
prosody's **increment over a lexical baseline**, never raw accuracy (M12); length/era/genre/typography
controlled at the corpus level (M5/M10/M13); and headline results independently re-derived by adversarial
verifier agents. Several findings are clean **negatives** — recorded as such.

## The findings

| # | What we tested | Result | Verdict |
|---|---|---|---|
| **F1** | Does any prosody signal exist? | great-vs-flat 0.984, great-vs-slop 0.930; length/vocab baselines at chance | Real signal exists |
| **F2** | Is it orthogonal to lexical slop, confound-controlled? | modhuman-vs-slop 0.89–0.90; great-vs-modhuman residual 0.653, **increment +0.102 over lexical (sig)** | Partly orthogonal; strong for human-vs-AI, weak for within-human quality |
| **F3** | Is the *order* of the features the missing piece? | order residualized against the histograms = 0.551 (chance), below the shuffle 0.567 | **KILL** (4-agent verified) |
| **F4** | Is our prosody *psychologically real*? (ZuCo EEG + eye-tracking) | reading-time prosody block p=0.0089 but via syllables/content; phrase-boundary + EEG channels **null** | Weak / qualified |
| **F5** | Is prosody a *paraphrase-robust* AI-detector? | multi-realization diff-in-drops **+0.052, CI [0.009, 0.098]** | Confirmed — for weak models |
| **F6** | Does it hold for a frontier model? | prosody flags GPT-4o at only **0.838** (vs 0.925 for weak models), pre-paraphrase | Frontier prose is already human-like |
| **Z2** | Does F5 generalize (2nd paraphraser + 2nd source)? | back-translation: +0.039 (ns); GPT-4o: **−0.008** | F5 is **conditional/narrow** |
| **F7** | Is *dictionary stress* the bottleneck? (speech-grounded prominence, GPU) | detection: ~no change; quality great-vs-modhuman 0.756→**0.825**, +0.069 **CI [0.000, 0.147]** | Split: no for detection, **borderline yes** for quality |

(Plus an exploratory probe: Paul Graham essays vs AI — prosody 0.96 but lexical 0.95, so prosody is **not
unique** there; it tracks register/genre, not quality.)

## What prosody IS and ISN'T (the cross-cutting conclusion)

- **IS:** a real, partly-lexical-orthogonal signal that separates human prose from *weaker* AI (strong) and
  tracks register/era/genre (strong). Its most robust, defensible property is **rhythmic variation** — good
  human prose varies; flat machine prose doesn't.
- **ISN'T:** a fine-grained within-human **quality** ranker (weak: ~+0.07–0.10 increment over lexical, and
  that contrast is itself era/register confounded), and **isn't** a durable AI-detector — its edge degrades
  as models get better (F6), vanishing for GPT-4o (Z2).
- **As a reward** (D3/D14): at most **one minor term** in a composite, weighted low, used as an anti-monotony
  regularizer, guarded against gaming (reward *variation* not regularity — M3; never optimize it alone —
  M2; diversity-preserving optimizer — D6). Not a standalone or dominant quality reward.

## What we ruled out vs what stays open

**Ruled out** (so these are not the explanation for the weak quality signal):
- The **classifier** is not the bottleneck (RF/SVM/seeds matched logreg — D14).
- The **order** of the features adds nothing over their histograms (F3, verified).
- **Dictionary citation-stress** is not the limiter for *detection* — a speech-grounded prominence front-end
  didn't help (F7); GPT-4o's human-like rhythm is genuine, not a measurement artifact.

**Still open** (the honest frontier):
1. **The fixed-formula paradigm.** Every pipeline ends in a *hand-designed* summary formula, even F7 (it made
   the per-unit values learned, then summarized them by fixed stats). We never let a model *learn* the
   prosody-sequence → answer mapping end-to-end. F7's borderline quality lift (+0.069) is the one hint that a
   richer use of the signal helps. Clean test = a learned sequence model on the prominence sequence — **blocked
   by data scale** (hundreds of passages overfit).
2. **The de-confounded, higher-power dataset.** The cleanest within-human quality contrast is still
   amateur-Reddit vs old-classics (register + era confounded) with wide CIs (~10–90 source groups). A
   within-author or within-genre/era quality gradient would sharpen every quality result — and is the data any
   learned model would need anyway.

Both open questions reduce to the **same wall: data**. That is the binding constraint, not the algorithm.

## Honest status

A coherent, well-verified, **negative-leaning but real** result: prosody is a genuine, weak, complementary
signal — strong on origin/register, weak on quality, fragile as AI improves. Two live threads could still
change the quality verdict (the F7 power-up; a learned representation on more data), both gated on a better
dataset. The project is at a natural write-up point: the story is complete and the open questions are clearly
bounded.

## If we wrote it up

The defensible claims: (1) text-prosody carries a real signal partly orthogonal to lexical slop, strongest as
human-vs-AI and as rhythmic-variation; (2) it does not finely rank within-human quality, and we ruled out the
classifier, the feature order, and (for detection) the per-unit extraction as the cause; (3) its AI-detection
value degrades for frontier models; (4) the methodological contribution — a confound-controlled,
pre-registered, group-level, adversarially-verified protocol for testing a candidate writing-quality signal,
including a clean way to fool yourself less (M11–M15). A clean negative with a sharp boundary is the result.
