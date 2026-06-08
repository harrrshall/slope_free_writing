# Prosody-as-Reward for Slop-Free Writing

An open-ended research project investigating whether the **rhythm of written prose** —
the cadence a reader hears internally while reading silently — can serve as a reward
signal for writing quality, complementing the lexical and semantic reward models used
in RLHF.

## The one-paragraph version

LLMs have surged at math and code because those domains have *verifiable rewards*. Writing
has no verifier, and the usual substitute — an LLM-as-judge — is biased toward the bland,
cliché-heavy "slop" we want to remove. This project hunts for writing-quality signals that
are real and hard to game. The first candidate is **text-derived prosody**: strong prose is
hypothesized to have characteristic rhythmic *variation* that flat or machine prose lacks,
and which lexical/semantic reward models cannot see. We test this cheaply first, then — only
if the signal is real — build it into a complementary reward term and into a diversity-preserving
preference-optimization pipeline.

## How to read this repo

Start with `AGENTS.md` (the operating manual), then:

- `docs/RESEARCH_GOAL.md` — what we are trying to prove and what would disprove it.
- `docs/BACKGROUND.md` — what is already known, and exactly where our contribution is new.
- `docs/METHODOLOGY.md` — the phased plan. **Phase 0 gates everything.**
- `docs/EXPERIMENT_LOG.md` — the running, dated record of work.
- `docs/FINDINGS.md` — validated conclusions.
- `docs/MISTAKES.md` — traps, so we never repeat them.
- `docs/DECISIONS.md` — why we chose what we chose.
- `docs/ENVIRONMENT.md` — setup, libraries, data.

## Status

Phase 0 (separation pre-test) — **complete: PROCEED** (2026-06-07). Text-prosody features separate
great/flat/slop well above chance (binary great-vs-flat 0.984, held-out 0.969; within-narrative
great-vs-slop 0.930, held-out 0.908; group-level permutation p=0.001), and the separation is not
explained by passage length or lexical richness (baselines at chance; survives residualization).
See `FINDINGS.md` F1.

Phase 1 (orthogonality H2 + genre/era control H3) — **complete: PROCEED, hardened** (2026-06-08).
Against a clean lexical-slop baseline, prosody is partly independent of lexical-slop detection:
binding great-vs-modhuman residual 0.653 (held-out 0.695, p=0.001), and modhuman-vs-slop (modern
human vs AI) 0.891 / held-out 0.904. Survives a Phase-1c hardening pass (typography normalized at
source, flat de-duplicated, permutation n=1000) and an independent 6-agent verification that
reproduced every core number. The strongest, most robust signal is rhythmic *variation*
distinguishing modern-human from AI prose. Per `DECISIONS.md` D14, prosody is NOT a standalone
quality reward; it is a candidate complementary / diversity (anti-monotony) signal. See `FINDINGS.md`
F2. Next: Phase 2 (test it as a diversity term) and run the same gauntlet on further candidate signals.

## Guiding principle

A clean negative result is a success. The goal is the truth about whether prose rhythm is
a usable reward signal, not a foregone "yes."
