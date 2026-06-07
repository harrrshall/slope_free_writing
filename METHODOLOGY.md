# Methodology

An open-ended, hypothesis-driven plan. Work the lowest-numbered phase whose **exit criteria**
are unmet. Do not skip. Each phase names what would let us proceed and what would kill the idea.
Update this file when findings change the plan.

---

## Phase 0 — Separation pre-test (the cheap kill-or-validate) — GATES EVERYTHING

**Question:** Do text-prosody features separate great prose from flat prose at all? (H1)

**Why first:** it is the cheapest experiment that can kill the whole idea. If prosody features
can't even separate obviously-different piles, nothing downstream is worth building.

**Steps:**
1. Assemble three small labeled piles (target ~50–100 passages each, matched length ~150–400 words):
   - **Great:** acknowledged-great human prose (Project Gutenberg — but mind era confound).
   - **Flat-human:** competent but unremarkable human prose (e.g. wire-service copy, manuals).
   - **AI-slop:** model outputs known to score high on the EQ-Bench Slop Score.
2. Extract prosody features per passage (see `docs/ENVIRONMENT.md` for tools):
   stress-pattern entropy/variation, syllables-per-word distribution, stress-interval statistics,
   sentence-length mean/variance, phrase/clause-boundary spacing, metrical-tension proxy.
3. Fit a simple, interpretable classifier (logistic regression / small tree) on these features
   only. Report accuracy, per-feature importance, and confusion across the three piles.
4. Eyeball: do the top features correspond to a plausible "cadence" story, or to a confound?

**Exit criteria (proceed):** prosody-only classifier separates the piles **significantly above
chance**, with at least one interpretable rhythm feature carrying weight.

**Kill criteria:** at/near chance after honest feature work → record negative result, log lesson,
go to Phase 4 (next candidate signal). Do not "rescue" with ever-more-baroque features.

> **STATUS: MET — PROCEED (2026-06-07).** See FINDINGS F1 / EXPERIMENT_LOG. Binary great-vs-flat
> balanced accuracy 0.984 (held-out 0.969), great-vs-slop 0.930 (held-out 0.908), group-level
> permutation p=0.001; baselines at chance; survives length+MATTR residualization; rhythm features
> carry the weight; sub-sentence-only great-vs-slop = 0.874. Caveats carried into Phase 1: great-vs-flat
> is genre/register-confounded and GREAT is pre-1928 (era). Phase 0 used logistic regression on Tier-A
> features only; the expensive prosodic metrical-tension (Tier B) was NOT needed (no danger-zone).

---

## Phase 1 — Orthogonality & confound control

**Questions:** Is the signal independent of lexical slop (H2)? Does it survive confounds (H3)?

**Steps:**
0. **(New, top priority from Phase 0 F1.) Break the genre/register and era confounds.** Phase 0's
   great-vs-flat (0.984) is fiction-vs-newswire; the cleaner contrast great-vs-slop was 0.930. So:
   (a) add a **competent MODERN-human fiction** pile (provenance-checked, not the euclaise band) and
   test great-vs-that and slop-vs-that — does prosody track *quality within narrative*, or just
   genre? (b) add a within-genre **era** control (modern acknowledged-great fiction) so GREAT is not
   100% pre-1928. If the signal collapses once genre+era are matched, that is the real kill test for
   "prosody tracks quality" (vs "prosody tracks genre/era").
1. Build a lexical-slop baseline classifier (reuse slop-forensics features).
2. Compare and **combine**: does prosody add accuracy *over* the lexical baseline (e.g. ablation,
   or prosody-residual after regressing out lexical features)?
3. Control for sentence length, vocabulary richness (MATTR/Flesch-Kincaid), era, and genre —
   by matching piles and/or including these as covariates. Re-test separation. (Length is already
   matched at the sampling stage per D9; vocabulary/MATTR residualization is already wired in.)

**Exit criteria:** prosody adds information beyond lexical features (H2) **and** separation
survives confound control (H3).

**Kill criteria:** signal fully explained by lexical features or by a confound → negative result, pivot.

---

## Phase 2 — Reward term

**Question:** Can the validated features become a stable scalar reward?

**Steps:**
1. Define a scalar `r_prosody` from the validated features. **Reward variation/cadence, not
   metrical regularity** (see AGENTS.md §5). Calibrate so a metronome does *not* score high.
2. Sanity-check on held-out passages: does `r_prosody` rank human-great > flat > slop?
3. Stress-test for trivial hacks (e.g. does padding with long words inflate it?).

**Exit criteria:** a documented, hack-resistant `r_prosody` that ranks held-out prose sensibly.

---

## Phase 3 — Integration into preference optimization (next milestone)

**Question:** Does the prosody reward help a policy without hurting coherence (H4)?

**Steps:**
1. Small open model. Combine `r_prosody` with a slop/quality reward as a **complementary** term
   (weighted sum or multi-objective; prosody never dominates).
2. Optimize with a **diversity-preserving** method (DivPO-style selection; forward-KL/JS per DPH-RL).
3. Evaluate: slop + repetition (objective), diversity (embedding spread, distinct-n), quality via a
   **held-out judge never optimized against** + small blind human pairwise. Headline plot:
   quality vs diversity, with and without the prosody term.

**Exit criteria:** prosody term improves quality/monotony without degrading coherence, vs ablation.

---

## Phase 4 — Next candidate signals (the open-ended branch)

This is research; the prosody bet may fail. Maintain a backlog of other "real, hard-to-game"
writing-quality signals to test with the *same* separation-first protocol — e.g. discourse/coherence
structure, narrative surprise/novelty (semantic distance), syntactic variety. Whatever survives
Phase 0–1 graduates toward the reward.

---

## Standing rules for every phase

- Cheapest-kill-first. Define the kill criterion before the run.
- Real human prose in every eval set. Held-out judge quarantined.
- Log every run in `EXPERIMENT_LOG.md`; distil only validated results into `FINDINGS.md`.
