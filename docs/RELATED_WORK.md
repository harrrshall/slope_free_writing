# Related Work — composite writing-quality reward + diversity-preserving RLHF

> Saved literature resource (deep-research, 2026-06-08, 17 primary sources, 18 claims verified 2-0 by
> adversarial batch verifiers, 0 refuted). **Read this instead of re-researching.** Extends `BACKGROUND.md`
> (which already had DivPO / DPH-RL / DiverseGRPO / LongWriter-Zero); the big additions here are **DARLING**
> (the closest prior art), **QEMPO**, and **GFlowNet fine-tuning**. Fast-moving area — re-check before publishing.

## Bottom line (the gap)

**No prior work combines a COMPOSITE hand-designed writing-quality reward (incl. a prosody/rhythm term) with a
diversity-preserving optimizer to reduce slop without collapsing diversity.** The two halves exist separately
and are mature (2024–2026); the **intersection is the novel slice** for this project.

## (1) Diversity-preserving optimizers — mature, but never wired to a composite prose reward

All motivated by: standard post-training / RLVR **sharpens the output distribution and collapses diversity**
(Pass@1 up, Pass@k down). Every one pairs with a **generic preference RM or a verifiable (math/SQL) reward**,
**never** a composite prose-quality signal — this is the clearest gap on the optimizer side.

- **DivPO** — Diverse Preference Optimization (arXiv:2501.18101, Lanchantin et al., Meta/NYU/ETH, Jan 2025).
  Selects chosen = rare-but-high-quality vs rejected = common-but-low-quality (diversity + a *single* quality
  measure). Big story-diversity gains at similar win rate. Reward = rarity-weighted quality, not multi-signal.
- **DPH-RL** — (arXiv:2509.07430). Identifies the **divergence choice** as the mechanism: forward-KL / JS are
  mass-covering (preserve diversity), reverse-KL is mode-seeking (kills it). Uses **only verifiable rewards**
  (math correctness, SQL execution) — no writing-quality/preference reward.
- **QEMPO** — (arXiv:2602.15894, Feb 2026). Closed-form entropy-maximizing solution under a quality
  constraint; proven entropy hierarchy QEMPO ≥ QEMPO-KL ≥ RLHF. Quality = a **generic preference RM**
  (UltraFeedback; eval with gpt-4o), no prosody/anti-slop term.
- **GFlowNet fine-tuning** — (arXiv:2310.04363, ICLR 2024; 2410.20147). Trains the LLM so its **sampling
  distribution ∝ reward** (distribution-matching) rather than reward-maximizing. Diversity-seeking by design.

## (2) Composite / multi-objective writing-quality rewards — exist, but with mode-seeking RL

- **LongWriter-Zero** — (arXiv:2506.18841). R1-Zero-style RL from a base model (no SFT, no synthetic data) for
  long-form **writing**, with a **composite multi-reward**: specialized RMs for **length control + writing
  quality + structural formatting**. Confirms a real writing-quality reward is used in RL. **BUT** paired with
  **standard GRPO-style (mode-seeking) RL — no mention of diversity / mode collapse / diversity preservation.**
  So even the strongest composite-prose-reward work does NOT do the combination.

## (3) Creative-writing / anti-slop metrics — WEAKEST-COVERED here (treat as unverified, not "absent")

The deep-research run under-covered this sub-area (no confirmed claim quantified these or showed them used as
RL rewards). Known to exist in the open-source creative-writing-eval niche; **needs a dedicated follow-up
search**. Seeds surfaced: EQ-Bench **Slop-Score** (eqbench.com/slop-score.html); anti-slop / creative-writing
detection (arXiv:2510.15061, 2508.21476, 2601.07149). The repo already uses slop-forensics/antislop (Sam Paech)
as the lexical baseline — see `BACKGROUND.md`.

## (Closest prior art) DARLING — the paper to beat / cite

- **DARLING** — "Jointly Reinforcing Diversity and Quality in Language Model Generations" (arXiv:2509.02534,
  Meta / facebookresearch/darling, Sep 2025). **Jointly optimizes a quality reward + a learned diversity signal
  (a learned partition function measuring diversity beyond surface lexical variation) in online RL**, beating
  quality-only RL on producing simultaneously higher-quality AND more novel outputs. **The single closest
  existing work.** Key distinction from this project: DARLING's quality = a **learned RM**, diversity = a
  **learned** partition function — NOT a hand-designed, interpretable composite (prosody + coherence + anti-slop).

## The novel slice (genuinely open) + how to position it

A **hand-designed, interpretable, multi-signal prose reward** (prosody/rhythm + discourse coherence + lexical
anti-slop) used as the **quality term inside a diversity-preserving optimizer** (DivPO pair-selection /
DPH-RL forward-KL / QEMPO entropy-constraint / GFlowNet distribution-matching). No surveyed paper does this.
The contribution is interpretability + the specific signal set, vs DARLING's learned black-box.

## Open questions (the crux to settle before committing)

1. **Does the prosody term add measurable signal ON TOP OF an LLM-judge / learned RM, or is it redundant once a
   strong quality RM is present?** Given our finding that prosody is *real-but-weak* (F2/F7), this incremental-
   value question is the crux of the whole contribution.
2. Can a hand-designed interpretable composite **match or beat DARLING's learned partition-function approach**
   on the quality/diversity Pareto frontier?
3. **Which diversity-preserving optimizer is the best host** for a composite prose reward (DivPO vs DPH-RL vs
   QEMPO vs GFlowNet)?
4. Is there 2024–2026 work specifically pairing **anti-slop rewards with diversity preservation** that fell
   outside this survey? (the weakest-covered area).

## Risks / why it might not be worth doing

- The novelty window is narrow: DARLING (Sep 2025) + QEMPO (Feb 2026) are very recent and both Meta — the field
  could close this gap any month.
- If prosody is redundant with a strong quality RM (open question 1), the "composite incl. prosody" framing
  loses its point; the contribution would collapse to "interpretable composite vs DARLING's learned signal."

## Citations (all arXiv primary unless noted)

| Work | id | what it is |
|---|---|---|
| DivPO | 2501.18101 | diversity-preserving DPO (rarity-weighted quality) |
| DPH-RL | 2509.07430 | forward-KL/JS mass-covering RL (verifiable rewards) |
| QEMPO | 2602.15894 | entropy-maximizing under quality constraint (Feb 2026) |
| GFlowNet fine-tuning | 2310.04363, 2410.20147 | sampling ∝ reward (distribution-matching) |
| LongWriter-Zero | 2506.18841 | composite writing reward (length/quality/format) + mode-seeking RL |
| **DARLING** | **2509.02534** | **closest: joint learned quality + learned diversity in online RL** |
| EQ-Bench Slop-Score | eqbench.com/slop-score.html | anti-slop metric (sub-area 3, under-verified) |
| anti-slop / creative metrics | 2510.15061, 2508.21476, 2601.07149 | creative-writing / slop detection (under-verified) |
| prosody-as-reward (speech only) | 2504.07532 | prosody RL reward — for TTS/speech, not written text |
