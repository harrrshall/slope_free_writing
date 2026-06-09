# Research Goal

## North Star (the big goal)

Find a **better reward model for writing quality** — one that can be used in RLHF /
preference optimization to produce **slop-free writing** *without collapsing the diversity
that good writing depends on*. Everything in this repo is in service of that.

Why it matters: reinforcement learning made models excellent at math and code because those
domains have cheap, automatic, unfakeable rewards. Writing has none. The reward is the
bottleneck, and a reward that is both honest and hard to game is the missing piece.

## Current milestone (the instant goal)

Establish whether **text-derived prosody features** ("prose music": stress-pattern variation,
syllabic/metrical cadence, phrase-boundary spacing, sentence-rhythm variance) carry a real,
*separable* signal of writing quality that lexical and semantic reward models miss — and, if
so, package it as a complementary reward term.

## Hypotheses

- **H1 (separation):** Text-prosody features distinguish acknowledged-great human prose from
  competent-but-flat prose (and from AI "slop") at significantly-above-chance accuracy.
- **H2 (orthogonality):** The prosody signal is at least partly *independent* of lexical slop
  features — i.e. it adds information a slop/lexical classifier does not already have.
- **H3 (no confound):** The separation in H1 survives controlling for obvious confounds
  (sentence length, vocabulary richness, publication era, genre).
- **H4 (reward utility — later):** Adding a prosody reward term to preference optimization
  improves human-judged quality and/or reduces monotony *without* degrading coherence,
  relative to the same setup without it.

## Success criteria

- **Milestone success:** H1, H2, and H3 all hold on a held-out set of *real human prose*, with
  effect sizes reported and confounds ruled out. (H4 is the next milestone, not this one.)
- **North Star success (long horizon):** a reward signal/model that measurably reduces slop and
  preserves diversity in a trained policy, validated against a held-out judge and blind human eval.

## Kill criteria (decide now, honor later)

- **Kill the prosody idea if:** after honest feature engineering, prosody features do **not**
  separate great from flat prose above chance (H1 fails), **or** the separation disappears once
  confounds are controlled (H3 fails), **or** the signal is fully explained by lexical slop
  features already in hand (H2 fails). Any of these → record the negative result in FINDINGS.md,
  write the lesson in MISTAKES.md, and pivot to the next candidate signal (see METHODOLOGY Phase 4).

## Scope and non-goals

- In scope: feature extraction from text, a separation study, and (if warranted) a complementary
  reward term and a small diversity-preserving preference-optimization experiment.
- **Out of scope:** building a "Nobel-laureate writer," pretraining from scratch, audio/TTS-based
  emotion scoring, and any claim that this single signal defines writing quality. It is one shard.
