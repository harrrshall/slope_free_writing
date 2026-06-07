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
  - ROBUSTNESS to the typography confound: residualizing out BOTH lexical-slop AND typography (quote
    glyphs), **modhuman-vs-slop survives at 0.713 (CI [0.612, 0.813])**; great-vs-modhuman weakens to
    0.606 (CI [0.522, 0.699], p=0.016); great-vs-slop 0.609 (CI touches chance).
- **Confidence**: medium. Every headline number was independently reproduced; the conclusion holds
  most strongly for modern-human-vs-AI, which survives lexical-slop + typography control.
- **Limits (found by independent adversarial verification)**:
  - **Typography confound**: modhuman uses different quote glyphs (`''`, spaced apostrophes) absent in
    great/modgreat; a typography-only classifier reaches 0.76–0.84. Prosody survives it for
    modhuman-vs-slop but only marginally for great-vs-modhuman. MUST normalize typography across classes.
  - **Word length**: mean_word_len differs by class and separates great-vs-flat at 0.93 alone; it was
    NOT in the Phase-0 residualizer (the Phase-1b slop baseline DOES include it, so the binding results
    control it, but the F1 confound story is incomplete).
  - The dominant prosody feature group by permutation importance is **syllable_dist, not sentence-length
    variation** (earlier wording was imprecise).
  - The FULL-lexical "kill" reflects a near-ceiling/circular baseline on great-vs-slop, not purely the
    diversity proxies (mechanism was overstated).
  - Minor: 2 byte-identical flat passages + ~6 flat near-dups straddle dev/held (affects flat held-out
    only, NOT the binding great/modhuman/slop contrasts); permutation p is resolution-floor-bounded;
    modgreat is small-n (8 books) and exploratory.
  - SLOP_BASELINE_COLS was chosen after seeing FULL kill the result (researcher d.o.f.); mitigated by
    confirming on the un-peeked held-out + new contrasts.
- **Implication**: H2 (orthogonality) and H3 (genre/era) are supported with caveats. Before
  publication-grade claims, do a Phase-1c hardening pass: normalize typography across all classes,
  de-duplicate flat, add mean_word_len to the F1 residualizer, raise permutation n to ≥1000, then
  re-confirm. The reward-term work (Phase 2) should lead with the modern-human-vs-AI contrast.

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
