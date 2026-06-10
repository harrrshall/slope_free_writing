# Prosody-as-Reward for Slop-Free Writing

An open-ended research project investigating whether the **rhythm of written prose** 
the cadence a reader hears internally while reading silently  can serve as a reward
signal for writing quality, complementing the lexical and semantic reward models used
in RLHF.

## The one-paragraph version

LLMs have surged at math and code because those domains have *verifiable rewards*. Writing
has no verifier, and the usual substitute  an LLM-as-judge  is biased toward the bland,
cliché-heavy "slop" we want to remove. This project hunts for writing-quality signals that
are real and hard to game. The first candidate is **text-derived prosody**: strong prose is
hypothesized to have characteristic rhythmic *variation* that flat or machine prose lacks,
and which lexical/semantic reward models cannot see. We test this cheaply first, then  only
if the signal is real  build it into a complementary reward term and into a diversity-preserving
preference-optimization pipeline.

## How to read this repo

Start with `AGENTS.md` (the operating manual), then:

- **`docs/SYNTHESIS.md`  the consolidated findings (F1–F7) and the honest bottom line. Read this first.**
- `docs/RESEARCH_GOAL.md`  what we are trying to prove and what would disprove it.
- `docs/BACKGROUND.md`  what is already known, and exactly where our contribution is new.
- `docs/RELATED_WORK.md`  saved literature survey (composite reward + diversity-preserving RLHF); read before re-researching.
- `docs/METHODOLOGY.md`  the phased plan. **Phase 0 gates everything.**
- `docs/EXPERIMENT_LOG.md`  the running, dated record of work.
- `docs/FINDINGS.md`  validated conclusions.
- `docs/MISTAKES.md`  traps, so we never repeat them.
- `docs/DECISIONS.md`  why we chose what we chose.
- `docs/ENVIRONMENT.md`  setup, libraries, data.
- `docs/EXTRACTION_UPGRADE_PLAN.md`  the current experimental arc.

## Repository layout

```
AGENTS.md  README.md  .gitignore  pytest.ini
docs/      all research markdown (goals, methodology, log, findings, decisions, …)
src/       shared libraries at root (features_lib, evaluate, …); drivers grouped in
  ├─ corpus/  extract/  experiments/  judge/  zuco/    run as `.venv/bin/python3 src/<group>/<name>.py`
tests/     pytest suite                run as `.venv/bin/python -m pytest`
data/      inputs: manifest.csv, slop_lists/, passages/, passages_raw/, texts/
results/   generated artifacts, grouped:
  ├─ features/   feature matrices (features_*.parquet, lexical_*.parquet)
  ├─ reports/    experiment outputs (report_*.json, heldout_report*.json)
  └─ splits/     group-aware quarantine splits (quarantine_split*.json)
```

## Status (2026-06-08)

Findings F1–F7 complete; full narrative in **`docs/SYNTHESIS.md`**. The honest bottom line:

- **Real but weak.** Text-prosody separates human prose from *weaker* AI (modhuman-vs-slop ~0.89–0.90) and
  tracks register/era; it is partly orthogonal to lexical slop (F2, increment +0.102, significant).
- **Not a quality ranker.** Within-human quality (great-vs-modhuman) is weak (~0.65 residualized). The
  *order* of the features adds nothing (F3, KILL, verified) and the classifier is not the bottleneck (D14).
- **Foundation only weakly confirmed.** Against real reading (ZuCo EEG + eye-tracking), our prosody predicts
  reading behaviour but via non-prosody-specific channels; phrase-boundary + EEG were null (F4).
- **A paraphrase-robust AI signal, but narrowly.** Prosody degrades less than lexical under paraphrase (F5,
  +0.052)  but only for weak models; it nearly vanishes for the frontier model GPT-4o (F6, Z2).
- **Better extraction barely helps.** Speech-grounded prominence (BERT trained on the Helsinki corpus) did not
  improve detection, and gave only a *borderline* lift to the quality contrast (F7, 0.756→0.825, CI grazes 0).

Per `DECISIONS.md` D14, prosody is at most **one minor, complementary term** in a composite reward
(anti-monotony / rhythmic variation), never a standalone quality reward. The two questions left open  a
*learned representation* vs hand-formulas, and a *de-confounded higher-power* quality dataset  both reduce to
the same binding constraint: **data**.

## Guiding principle

A clean negative result is a success. The goal is the truth about whether prose rhythm is
a usable reward signal, not a foregone "yes."
