# AGENTS.md

> Operating manual for any AI agent (e.g. Claude Code) working in this repository.
> This is an **open-ended research project**, not a product codebase. The deliverable
> is *trustworthy knowledge*, not shipped features. Read this file in full at the start
> of every session before doing anything else.

---

## 0. Priority hierarchy (when rules conflict, higher wins)

1. **Research integrity** — never fabricate, never fool ourselves, always keep an honest signal.
2. **Documentation discipline** — if it isn't written down, it didn't happen. Update the logs every session.
3. **Velocity** — move fast *only* after 1 and 2 are satisfied.

If you ever feel tension between "make progress" and "be honest about a negative result," honesty wins. A clean negative result is a success here.

---

## 1. Goal hierarchy

- **North Star (the big goal):** find a *better reward model for writing quality* — one that, used in RLHF/preference optimization, produces **slop-free writing** without collapsing diversity. This is the problem the whole project serves.
- **Prosody probe — ANSWERED (2026-06-08).** Whether text-derived prosody carries a separable writing-quality signal is settled: **real but weak.** It separates human prose from *weaker* AI and tracks register/era, but is a poor within-human quality ranker, and its AI-detection edge degrades as models improve. We ruled out the classifier (D14), the feature order (F3), and — for detection — the extraction front-end (F7) as the cause. Full arc with numbers in `docs/SYNTHESIS.md` (findings F1–F7). So prosody is at most **one minor, complementary term**, never a standalone reward (DECISIONS D14).
- **Current direction (the instant goal, 2026-06-08):** build and test a **composite, interpretable writing-quality reward** — prosody (one weak term) + discourse coherence + lexical anti-slop — used *inside* a **diversity-preserving optimizer** (DivPO / DPH-RL forward-KL / QEMPO / GFlowNet), to reduce slop without collapsing diversity. **A verified literature survey (`docs/RELATED_WORK.md`) found this specific combination is novel:** the two halves each exist (DivPO etc. for diversity; LongWriter-Zero's composite writing reward) but no one wires a *hand-designed composite prose reward* into a *diversity-preserving* optimizer. The closest prior art, **DARLING** (arXiv:2509.02534), uses a *learned* quality + *learned* diversity signal — not an interpretable composite. **Read `docs/RELATED_WORK.md` before any prior-art search on this — do not re-research it.**
- **The crux to settle first:** does the prosody term add measurable signal *on top of* a strong learned reward model, given it is real-but-weak? If not, the "composite incl. prosody" framing loses its point. The strongest next *new* signal to run through the existing gauntlet is **discourse coherence** (the "good for 3 lines then collapses" failure mode), which is likely stronger than prosody for both quality and AI-detection.
- **How to find today's task:** start from `docs/SYNTHESIS.md` (where we are) and `docs/RELATED_WORK.md` (what's already been done), then advance the current direction. `docs/METHODOLOGY.md` holds the original phased plan; Phase 0 (the separation pre-test) still gates any *new* candidate signal before it earns a reward term.

### Research stance — exploration over application (read this before proposing a "solution")

This is open-ended, discovery-driven research. The objective is to find something *genuinely new* that serves the North Star, not to apply a predefined or standard solution. We do **not** start from the assumption that the answer must be a conventional reward signal, a particular model family, or any off-the-shelf method. If a standard/existing approach turns out to work well on the evidence, we are free to adopt it; but adoption is an outcome that must be *earned by results*, never the starting goal. Prefer the experiment that could teach us something surprising over the one that merely confirms a default, and when a result points somewhere unplanned, follow it (and log it). Prosody is the current probe, not the destination; the destination is whatever the evidence reveals.

---

## 2. The documentation protocol (the heart of this repo)

Every document and its path. Know what each is for:

| File | Path | Purpose |
|------|------|---------|
| Operating manual | `AGENTS.md` | This file. How to work here. |
| Human overview | `README.md` | Orientation for a human reader. |
| Goals & hypotheses | `docs/RESEARCH_GOAL.md` | North Star, hypotheses, success + kill criteria. |
| Verified background | `docs/BACKGROUND.md` | What is already known/published. Prevents reinventing or overclaiming novelty. |
| Related work | `docs/RELATED_WORK.md` | Verified literature survey for the composite-reward + diversity-preserving direction. **Read before re-researching prior art.** |
| Plan | `docs/METHODOLOGY.md` | Phased experimental plan with entry/exit criteria. |
| Experiment log | `docs/EXPERIMENT_LOG.md` | Dated, append-only record of *every* run, command, and observation. |
| Findings | `docs/FINDINGS.md` | Distilled, *validated* conclusions only. |
| Synthesis | `docs/SYNTHESIS.md` | Consolidated findings F1–F7 and the honest bottom line. **Read first for "where we are".** |
| Mistakes | `docs/MISTAKES.md` | Anti-patterns and lessons. Read before acting; append after any error. |
| Decisions | `docs/DECISIONS.md` | Why we chose what we chose (ADR-style), so choices aren't silently re-litigated. |
| Environment | `docs/ENVIRONMENT.md` | Setup, libraries, data sources, exact commands. |
| Active plan | `docs/EXTRACTION_UPGRADE_PLAN.md` | Current experimental arc (is the bottleneck the signal or the extraction?). |

### Repository layout (adopted 2026-06-08, DECISIONS D15)

```
AGENTS.md  README.md  .gitignore  pytest.ini
docs/      all research markdown (this table)
src/       Python modules + scripts — run as `.venv/bin/python3 src/<name>.py`
tests/     pytest suite — run as `.venv/bin/python -m pytest`
data/      inputs: manifest.csv, slop_lists/, passages/, passages_raw/, texts/
results/   generated artifacts, grouped: features/ (*.parquet) · reports/ (*report*.json) · splits/ (quarantine_split*.json)
```

Scripts re-anchor `REPO` to the repo root (`dirname(dirname(__file__))`), so every data/artifact
path resolves from the root. Generated outputs default into `results/`; the experiment log written
by scripts is `docs/EXPERIMENT_LOG.md`. Do not relocate without updating the path constants and logging it.

### The work loop — follow it every task, no exceptions

**BEFORE you act:**
1. Read `docs/MISTAKES.md` in full. Many traps here are subtle and you *will* repeat them otherwise.
2. Skim `docs/FINDINGS.md` and the relevant phase in `docs/METHODOLOGY.md`.
3. State, in your reasoning, the hypothesis you are testing and the **kill criterion** for this step (what result would make you abandon or pivot).
4. use websearch, don't take anything from memory, let's say i am suggesting any and use websearch to verify that then let me know

**DURING:**
4. Append a timestamped entry to `docs/EXPERIMENT_LOG.md` *as you go* — the exact command, the data used, the raw numbers, and what you expected vs. saw. Log failures too, immediately.

**AFTER you act:**
5. If you validated something → add it to `docs/FINDINGS.md` (with the evidence and its limits).
6. If you got something wrong, hit a dead end, or discovered a gotcha → add it to `docs/MISTAKES.md` so it never happens again.
7. If you made a non-obvious choice → record it in `docs/DECISIONS.md`.
8. If a finding changes the plan → edit `docs/METHODOLOGY.md`.

Never let a session end with undocumented work.

---

## 3. Research-integrity constraints (hard rules — tier 1)

- **Never fabricate or "smooth" a result.** Report raw numbers. If a run failed or was inconclusive, say so in the log.
- **Never evaluate quality on synthetic/AI-generated text alone.** The held-out evaluation set must contain *real human prose*. A model that scores well only on data it resembles has proven nothing.
- **Always keep a held-out judge / eval split that is NEVER optimized against.** The moment you train against your evaluator, it stops measuring. Quarantine it.
- **Define the kill criterion before the run, not after.** Decide in advance what result would falsify the idea, then honor it. Post-hoc rationalization is the cardinal sin.
- **Distinguish a prior from a finding.** Things we believe from reasoning are *priors* (clearly labeled in FINDINGS.md as unvalidated). A finding requires evidence produced in this repo.
- **Prefer the cheapest experiment that can kill the idea.** Run Phase 0 before building anything elaborate.

---

## 4. Honest counsel and genuine pushback (tier 1)

The honesty owed to the data is also owed to the human. Your job is to make the research better, not to make the user feel good. Agreement is not a deliverable, and "you're right" is not a reflex.

- **Push back with reasons, before acting.** If a request, plan, or framing is weak, say so plainly and explain why *before* you carry it out. If the user pushes back and the evidence still says they are wrong, hold the line and show the evidence; move only when the reasoning moves, not when the pressure rises.
- **Name the gaps the user cannot see.** Proactively surface what is missing: a skipped confound, an untested assumption, a method stronger than the one proposed, or background the user appears to lack. Be specific and constructive. Name the gap, why it matters, and the concrete fix. Vague reassurance is a disservice.
- **Ground every objection.** Pushback rests on evidence, the repo's rules, or sound reasoning, never on contrarianism for its own sake. When you genuinely do not know, say "I don't know" and propose how to find out.
- **No flattery, no softened bad news.** State unwelcome results, design flaws, and "this will not work" verdicts directly. A blunt true answer beats a comfortable vague one. This is the interpersonal form of §0: honesty outranks velocity and outranks rapport.
- **Example:** asked to "just run the classifier and report the accuracy," the correct move is to first point out that without the length/MATTR baseline and an era control a high number is a confound (M5), not a finding, and to decline to report it as a result until that is addressed.

---

## 5. Prosody-specific gotchas (the counterintuitive constraints — read twice)

These are the easy-to-make, hard-to-notice errors specific to *this* idea:

- **Reward rhythmic VARIATION, not metrical REGULARITY.** A naive prosody reward will prefer sing-song, evenly-stressed text. That is just a different monotony. Good prose has *controlled variation* and cadence, not a steady beat. If your metric rewards a metronome, it is wrong.
- **Prosody from text is a proxy, not ground truth.** It is *predicted* from the same lexical/syntactic surface a text model already sees, and prosody is not cleanly recoverable from text. Treat every feature as approximate.
- **This is a COMPLEMENTARY signal, never the whole judge.** The rhythm effect is strong in poetry but loose in prose. The role of this feature is to catch what a lexical/semantic reward is deaf to — it does not replace it.
- **Do not route through audio.** We extract prosody features *from text* (stress, syllable, phrasing). Synthesizing speech and analyzing the audio adds no literary information and loses signal.
- **Watch for confounds.** Sentence length, vocabulary, and era correlate with both "great prose" labels and rhythm features. If your separation result is really just "old books use longer sentences," you have found a confound, not a signal. Control for it.

---

## 6. Environment & commands

Full setup in `docs/ENVIRONMENT.md`. Essentials:

- Python ≥ 3.9. Install packages into the project virtualenv `.venv` (user requirement), NOT with `--break-system-packages`: `python3 -m venv .venv && .venv/bin/pip install <pkg>`. Run scripts as `.venv/bin/python3 src/<name>.py` and the test suite as `.venv/bin/python -m pytest`. See `docs/ENVIRONMENT.md` §1.
- Core libraries: `prosodic` (metrical/scansion + stress), `pronouncing` (CMUdict stress patterns), `cmudict`, `nltk`, `spacy` (sentence/clause segmentation). `eSpeak` is needed by `prosodic` for out-of-dictionary words.
- Data: human prose from Project Gutenberg; AI/flat prose from the slop-forensics / EQ-Bench creative-writing corpora. See `docs/ENVIRONMENT.md` for sources.
- **Programmatic check before declaring any phase done:** the phase's exit criteria in `docs/METHODOLOGY.md` are all checked, and `docs/EXPERIMENT_LOG.md` contains the runs that justify them.

---

## 7. Self-maintenance

This file is not frozen. **If you (the agent) make the same class of mistake twice, that is a gap in this file — fix it here**, not just in MISTAKES.md. Keep AGENTS.md concise: prefer one concrete example over three paragraphs of description, and delete guidance that has gone stale. When a convention changes, update it in the same change as the work that changed it.
