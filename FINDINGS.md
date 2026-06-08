# Findings

Only **validated** conclusions go here — things supported by evidence produced in this repo,
with the evidence and its limits stated. Keep priors (beliefs from reasoning, not yet tested)
in the clearly-marked section at the bottom, and promote them up only once evidence exists.

## Validated findings

### F1: Text-prosody features separate great / flat / slop prose well above chance, and not via length or vocabulary (H1 supported)
- **Evidence** (EXPERIMENT_LOG 2026-06-07; report_A.json, heldout_report.json): Tier-A text-prosody
  features (stress timing, syllable distribution, clause spacing, sentence-length variation), with a
  simple logistic-regression + StandardScaler in group-aware 5-fold CV (group = source book /
  Reuters category / model), give:
  - binary great-vs-flat balanced accuracy **0.984** (CI 0.958–1.000), group-level permutation
    **p=0.001**; 3-class **0.917**, p=0.001.
  - length-only and MATTR-only baselines sit at **chance** (0.511 / 0.536); prosody beats the best
    baseline by **+0.448**; fold-internal residualization (regressing out sentence-length-mean +
    MATTR per train fold) still gives **0.961** (CI 0.924–0.992).
  - the within-narrative QUALITY contrast great-vs-slop (both narrative prose) = **0.930**, and
    **0.874 using sub-sentence rhythm features only** (no sentence-length, no OOV/vocab).
  - quarantined held-out, scored once: great-vs-flat **0.969**, great-vs-slop **0.908** — matches dev.
  - direction is correct: great prose has ~5.5× the sentence-length variance of slop (243 vs 44) and
    higher stress run-length variance — rhythmic VARIATION, not metrical regularity (M3 guard passed).
- **Confidence**: medium-high that a real, separable text-prosody signal exists that is not explained
  by passage length or lexical richness. Pre-registered kill criterion (frozen before any score) was
  met on all six PROCEED conditions with no rescue round.
- **Limits (what it does NOT show)**:
  - great-vs-flat (0.984) is **genre/register confounded** (pre-1928 fiction vs 1987 finance wire);
    the cleaner quality signal is great-vs-slop (0.91–0.93), still strong.
  - **era confound** remains: GREAT is pre-1928 only (user choice); the euclaise era-band is
    narrative-vs-wire, not a within-genre era control.
  - AI-slop's documented sentence-length monotony may make this contrast comparatively easy; it is
    NOT yet shown that prosody separates great from *competent modern human* prose.
  - sentence-length variation + syllable distribution dominate; deeper sub-sentence rhythm is present
    (0.874) but should be probed further.
  - H2 (orthogonality to lexical slop) and H3 (full confound control incl. genre/era) are NOT yet
    established — that is the Phase 1 job.
- **Implication**: the prosody bet survives the cheap kill test. Proceed to Phase 1, whose first
  priority is controlling genre/register and era (great-vs-slop and modern-human contrasts), then
  testing orthogonality to a lexical-slop baseline.

Format for each finding:
```
### F#: <one-line claim>
- Evidence: <which log entry / run, numbers, effect size>
- Confidence: <low/med/high> and why
- Limits: <what it does NOT show; confounds not yet ruled out>
- Implication: <what changes because of this>
```

### F2: Text-prosody is PARTLY independent of lexical-slop features; strongest for modern-human vs AI (H2/H3 supported, with caveats)
- **Evidence** (Phase 1b; report_p1.json, heldout_report_p1.json; ALL core numbers independently
  re-derived from scratch by 6 adversarial verifier agents + a typography-control re-run):
  - The original "prosody vs FULL-lexical" residual KILLED on great-vs-slop (0.56) — but that was an
    artifact of putting diversity/variation-proxy features (distinct-n, MATTR) in the lexical baseline.
    Against a genuine SLOP-ONLY baseline (slop-word/n-gram densities + readability incl. word length),
    prosody's residual survives: great-vs-slop 0.695, modhuman-vs-slop 0.752, binding great-vs-modhuman
    **0.669 dev / 0.726 held-out** (group-perm p≈0.002–0.005).
  - H3-A (genre+era matched): modhuman-vs-slop prosody-only **0.891 dev / 0.904 held-out** (p≈0.001).
  - HARDENED (Phase 1c): typography normalized UNIFORMLY across all piles AT SOURCE (the confound's
    correct fix; a typography-only classifier drops 0.758->0.481 on great-vs-modhuman, prosody features
    unchanged), flat de-duplicated, perm n=1000. On this cleaned corpus the binding contrasts HOLD:
    **great-vs-modhuman resid-vs-SLOP = 0.653 (CI [0.571,0.732], p=0.001), held-out 0.695**; increment of
    prosody over the slop baseline = **+0.102 (CI [0.034,0.168], now significant**, was +0.016 ns);
    **modhuman-vs-slop = 0.891 dev / 0.904 held-out (p=0.001)**. (NB: residualizing typography as a
    *covariate* over-removed prosody variance to 0.606/p=0.016 in Phase 1b; normalizing at source is the
    correct method and the signal survives it.)
- **Confidence**: medium-high. Every headline number was independently reproduced by 6 adversarial
  verifiers, and the binding results survived the hardening pass (typography removed at source, flat
  deduped, n=1000) with held-out confirmation. Strongest for modern-human-vs-AI.
- **Limits**:
  - RESOLVED in Phase 1c: typography confound (normalized at source, classifier 0.76->0.48), flat
    dev/held dedup (14 near-dup Reuters passages removed; binding contrasts unaffected), word length
    (in the slop baseline the binding residual controls it), permutation n (=1000).
  - The dominant prosody feature group by permutation importance is **syllable_dist, not sentence-length
    variation** (earlier wording was imprecise; corrected per M14).
  - The FULL-lexical "kill" reflects a near-ceiling/circular baseline on great-vs-slop (the slop lists
    were built from AI text), not purely the diversity proxies (mechanism was overstated per M14).
  - This is human-vs-AI separation + a within-human quality gradient; it is **not** a demonstration that
    prosody ranks fine-grained writing quality. great-vs-flat (0.98) is genre-confounded (inherent).
  - modgreat is small-n (8 books, exploratory); the within-genre era null holds at the sub-sentence level.
  - SLOP_BASELINE_COLS was chosen after FULL killed the result (researcher d.o.f.); mitigated by
    pre-registered held-out confirmation on the cleaned, un-peeked corpus.
- **Implication**: H2 (orthogonality) and H3 (genre/era) are supported and survive the hardening pass.
  Per D14, prosody is NOT a standalone quality reward; the strongest, most robust signal is rhythmic
  *variation* distinguishing modern-human from AI prose. The natural next step is to test it as a
  **complementary diversity / anti-monotony term** (lead with the modhuman-vs-slop contrast), and to run
  the same confound-controlled gauntlet on the next candidate signals (narrative surprise, discourse
  coherence, syntactic variety).

---

## Priors (NOT yet validated — do not cite as results)

These come from reasoning and prior literature, not from experiments in this repo. They are
hypotheses to test, listed so we don't mistake them for conclusions.

- **P1:** Prose has measurable implicit-prosody/rhythm (well-supported in cognitive science), but
  the effect in *prose* is subtle/loose compared to poetry — so any reward signal is likely weak
  and complementary, not dominant.
- **P2:** A naive prosody reward will tend to reward metrical *regularity*, which is just another
  monotony; the real target is controlled *variation*. Must be guarded against in Phase 2.
- **P3:** Lexical slop and prosodic cadence are *probably* partly independent, but this is unproven —
  Phase 1 must establish it (H2), not assume it.
- **P4:** Era/genre/sentence-length are likely confounds in any "great vs flat" prose comparison.
- **P5:** Even if all of this works, it is one shard of the North Star; the dominant reward will
  still be lexical/semantic. Prosody's job is to catch what those miss.
