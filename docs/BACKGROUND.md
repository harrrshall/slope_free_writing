# Background — what is already known (and where we are new)

> Purpose: keep us honest. This prevents reinventing existing methods and prevents
> overclaiming novelty in front of reviewers. Everything below was verified via literature
> search; update with citations as you read more deeply. **Do not claim novelty for anything
> in the "already done" section.**

## The reward problem (why writing lags math/code)

- Math/code advanced under RL because their rewards are **verifiable** — automatic, dense,
  unfakeable (test suites, exact-match, proof checkers). Writing has no such verifier.
- The substitute, **LLM-as-judge**, is biased toward slop: judges reward overwrought phrasing
  and "vocab maxxing" that *degrade* writing quality but inflate the score (documented by the
  EQ-Bench maintainers). Optimizing against such a judge amplifies slop (Goodhart).

## Slop is partly measurable (use it, don't reinvent it)

- EQ-Bench's **Slop Score** and the **slop-forensics / antislop** toolkit (Sam Paech) quantify
  GPT-isms: over-represented words, "not-X-but-Y" patterns, over-used trigrams, plus style metrics
  (lexical diversity / MATTR, sentence/paragraph length). Reuse these as the lexical baseline.
  Repos: github.com/sam-paech/antislop-sampler, github.com/sam-paech/slop-forensics, EQ-bench/creative-writing-bench.

## Diversity collapse and its mitigations — ACTIVE AND CROWDED (already done)

This whole area is hot as of 2025–2026. We are **not** the first to attack it. Known work:

- **Mode collapse under RLHF is well-documented** (Kirk et al. 2024 show RLHF reduces diversity
  vs SFT; reverse-KL is mode-seeking; social-choice analysis shows RLHF over-represents the
  majority preference).
- **DivPO — Diverse Preference Optimization** (Lanchantin, A. Chen, Dhuliawala, P. Yu, Weston,
  Sukhbaatar, Kulikov; Meta/NYU/ETH; arXiv:2501.18101): selects rare-but-high-quality outputs as
  *chosen*, common-but-low-quality as *rejected*; reports large story-diversity gains at similar
  win rate. This is the core machinery for the later policy stage.
- **CPO — Creative Preference Optimization**: injects explicit creative dimensions into the objective.
- **DiverseGRPO** (arXiv:2512.21514): diversity-aware GRPO with a semantic-clustering creativity
  bonus; establishes a quality–diversity Pareto frontier (for images).
- **DPH-RL** (OpenReview xPEsxcO7F7): uses mass-covering f-divergences (forward-KL / JS) to prevent
  diversity collapse — i.e. the "use forward-KL not reverse-KL" lever, already published.
- **LongWriter-Zero** (arXiv:2506.18841): RL for long-form writing with a composite reward
  (writing RM + length + format). "Multi-component writing reward + RL" is already done.
- A 2606.xxxx mechanistic taxonomy of RLHF failure (reward hacking, collapse, evaluator gaming,
  two external judges) closely resembles our diagnostic eval design — cite it, don't duplicate it.

## Prosody-as-reward — exists ONLY for speech (this is our gap)

- Prosody has been used as an RL reward, but every instance is for **speech/audio**:
  preference-guided prosody learning in TTS (arXiv:2509.18531), multi-reward GRPO with an
  LLM-annotated prosody/rhythm reward in TTS (arXiv:2511.21270), prosody as a teaching signal from
  human *voice* feedback (ICMI 2024), prosody-aware RL for speech emotion (EmotionThinker, 2026).
- **No located work uses the *implicit prosody of written text* as a reward for *written* writing
  quality.** That specific wiring is our contribution. (Re-check periodically; the field moves.)

## Cognitive grounding (why this isn't a fantasy)

- **Implicit Prosody Hypothesis** (Fodor 2002; Breen et al.): silent readers generate an internal
  prosodic/rhythmic representation that shapes interpretation, with behavioral, eye-tracking, and
  EEG/ERP evidence. So "prose has music even unread aloud" is a measurable phenomenon.
- **Caveat from the same literature:** metrical anomalies strongly disrupt *poetry* reading but
  affect *prose* far less — i.e. the prose rhythm signal is real but **subtle and loose**.
  This is exactly why we treat it as complementary, not primary.

## Tooling precedent

- The **Prosodic** library (Heuser, Falk, Anttila; github.com/quadrismegistus/prosodic) parses
  text → line → word → syllable → phoneme and runs a constraint-based metrical parser.
- It has already been applied to **prose cadence** (a "Metrical Tension Score" computed over
  speeches and procedural texts, not just verse) — precedent that prose-rhythm features are
  extractable and meaningful.

## Our honest delta (state it this way to reviewers)

The diversity-collapse problem and its fixes are known and active (cite DivPO, DPH-RL, DiverseGRPO,
LongWriter-Zero). Slop measurement exists (cite EQ-Bench). **What is new here is (1) a prose-rhythm /
implicit-prosody reward signal for *written text*, which appears unexplored, and (2) the specific
synthesis** — a slop-resistant + prosody-aware reward optimized with diversity-preserving methods,
applied to creative writing. We adapt known methods to a new domain and test one genuinely new signal.
