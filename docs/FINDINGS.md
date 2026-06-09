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

### F3: Order-aware (sequence) hand features add NOTHING beyond the order-blind summary stats on the clean within-human contrast — a clean NEGATIVE (the ~0.65 ceiling is the signal/series, not order-blindness)
- **Question**: the within-human quality contrast (great-vs-modhuman) capped at ~0.65 with 14 order-BLIND
  summary stats. Was the cap caused by throwing away ORDER (sl_var knows there is variation but not the
  long-short-long-short vs long-long-short-short *pattern*)? Cheapest-kill-first test (M9) before any
  Helsinki/CWT/learned-rep investment: 10 order-aware features (autocorrelation, Mann-Kendall trend,
  between-thirds variance ratio, low-freq fraction, turning-point rate, runs-z) on the SAME SL/SPW/SYMS/PHG
  series, pre-registered (EXPERIMENT_LOG 2026-06-08), judged on great-vs-modhuman through the F2 gauntlet.
- **Evidence** (report_stage1.json; report_stage1_verification.json; independently re-derived to 6 dp by 4
  adversarial verifiers, workflow wzs5gdntm): all three pre-registered gates FAIL.
  - PRIMARY-1 (lift over histograms): ba(summary+order) − ba(summary) = **−0.016**, CI[−0.078, +0.042];
    order features slightly *hurt* (overfit). Summary-only ba=0.787 (raw prosody, > the 0.653 slop-residual).
  - PRIMARY-2 (order REAL vs SHUFFLED, decisive): single-shuffle +0.015 CI[−0.082, +0.119]; the robust
    40-seed redo = **+0.027 CI[−0.072, +0.129]** (floor far below 0).
  - PRIMARY-3 (orthogonality): order-augmented increment-over-slop **+0.031** CI[−0.049, +0.112] vs the
    summary-only **+0.102** bar — order *lowers* the orthogonal increment.
  - **Decisive**: order residualized against the summary histograms = **0.551** (chance), *below* the
    residualized shuffle (0.567) → the order information is fully redundant with the histograms.
  - Rescue exhausted: no model (logreg/RF/GB/HGB/interactions), CV seed (PRIMARY-1 negative 10/10), or
    feature subset clears a CI lower bound > 0; per-feature gaps (sl_bvr p=0.009, phg_ac1 p=0.018) do not
    survive Bonferroni. No false-kill bug (series reconstruction 0.0 abs-diff; shuffle genuine; held-out untouched).
- **Confidence**: high. Reproduced exactly by 4 independent verifiers; robust to model family, CV seed,
  feature cherry-picking, and the shuffle-seed lottery.
- **Limits (what it does NOT show)**:
  - This kills ORDER of the *existing* dictionary-stress / sentence-length series. It does NOT show those
    series' VALUES are good or bad, and does NOT test a BETTER-extracted per-unit signal (speech-grounded
    prominence, Helsinki/CWT) — that label-quality axis is still open.
  - Real order features DO carry a *tiny* genuine signal vs order-destroyed series (+0.027 point, p<0.001
    over 40 seeds), but it is swamped by group-level CIs and redundant with the histograms.
  - Power is the binding limit: 70 source groups + n=127 give CIs ~±0.10, too wide to resolve a ~0.03 effect;
    the contrast itself is still amateur-Reddit vs old-classics (register/era confounded).
- **Implication**: per M9, do NOT build CWT / learned-sequence models on these hand series — a richer ORDER
  encoder on the same series will not clear the group-level noise. Redirect the next lever to (a) better
  per-unit extraction VALUES (Helsinki speech-grounded prominence, Stage 2), (b) a learned representation
  off raw text (D14 ceiling probe), and/or (c) a higher-ceiling, higher-power within-genre/era dataset. See
  DECISIONS D16, MISTAKES M15.

### F4: Our text-prosodic PRIMITIVES weakly predict real silent-reading behaviour beyond length/freq/surprisal — but via the least prosody-specific channel; the distinctive boundary + EEG signals are NULL (ZuCo validation oracle)
- **Question/method** (Track Z, pre-registered): do our per-word prosodic primitives (CMUdict syllables `nsyl`,
  content-vs-function `is_content`, clause/sentence `is_boundary`) predict ZuCo 2.0 reading behaviour (8 subjects,
  48,997 words, 349 Wikipedia NR sentences) BEYOND word length, log-frequency (wordfreq), GPT-2 surprisal,
  position, and sentence-final, with subject fixed effects and SENTENCE-clustered SE; Wald F-test on the prosody block.
- **Evidence** (results/report_zuco.json; data/zuco/features_zuco.parquet):
  - SKIP (was the word fixated): prosody block F=100.9, p=1e-21 — strong, but driven by `is_content` (+0.24,
    function words skipped), a generic lexical-class eye-movement effect, not distinctively rhythmic.
  - READING TIME (log TRT | fixated): prosody block F=11.6, **p=0.0089** — clears the pre-registered p<0.01 bar but
    barely; carried by `nsyl`/`is_content`; `is_boundary` (the wrap-up effect, the cleanest prosody-specific
    prediction) is NULL (coef +0.002, p=0.66).
  - THETA EEG: prosody block p=0.42 — NULL.
- **Confidence**: low-medium. Confirms the primitives carry REAL signal readers respond to (not noise), but does
  NOT show the extraction captures a strong, distinctively prosodic implicit-rhythm signal.
- **Limits**: ZuCo NR is register-homogeneous Wikipedia text (low prosodic variation across sentences) — a weak
  test by construction; 8 subjects; word length and syllable count are collinear; boundary is punctuation-based.
- **Implication**: the P1 foundation (text-prosody corresponds to what readers actually do) is WEAKLY supported,
  not strongly. With F3/D14, the prime suspect remains the per-unit VALUES (dictionary stress is crude); a richer
  prominence signal might predict reading better AND rank quality. Does not change the critical path (de-confounded
  within-author dataset + values/prominence test).

### F5: Prosody is significantly MORE robust to paraphrase than lexical — P6 CONFIRMED (multi-realization). The project's one positive, unique-prosody result: paraphrase-robust origin/AI detection
- **Method** (Track A, pre-registered twice): paraphrase the 160 modhuman+slop passages (meaning + length preserved,
  Claude, symmetric), train each detector on ORIGINAL features, score paraphrased, measure the accuracy DROP (paired,
  GROUP-bootstrap over source_id, M11). Binding = difference-in-drops (lexical − prosody), CI lower bound > 0.
- **Two stages**:
  - SINGLE realization (report_paraphrase.json): diff-in-drops **+0.056, CI [−0.003, 0.115]** → NEAR-MISS, CI grazes 0.
    Diagnosed as the M15 mistake (one paraphrase realization is underpowered) + the AI-side ceiling (slop = 15
    generator-model groups, lars1234's full count — verified — so the CI is group-limited, NOT passage-limited).
  - MULTI-realization power-up (report_paraphrase_multi.json): 3 independent paraphrase realizations of the SAME 160,
    averaged, folded into the group bootstrap (pre-registered, M11 kept, no goalpost move). PROSODY 0.925→0.888
    (drop **+0.038**, its own CI grazes 0 → nearly paraphrase-INVARIANT); LEXICAL 0.931→0.842 (drop **+0.090**,
    CI [0.054, 0.124] → clearly degrades). Difference-in-drops **+0.052, CI [0.009, 0.098]** → **P6-SUPPORTED**.
- **Verdict** (per pre-reg, HONORED, M7): **P6-SUPPORTED**. Lexical AI-detection degrades ~2.4× as much as prosody
  under paraphrase; prosody is nearly paraphrase-invariant. The point estimate was stable across stages (+0.056→+0.052);
  multi-realization legitimately reduced the paraphrase-lottery variance (M15), tightening the CI to exclude 0 — not p-hacking.
- **Confidence**: medium. A real, pre-registered positive, but a MODEST margin (CI floor +0.009, close to 0).
  Mechanistically sound: paraphrase rewrites the slop n-grams (lexical collapses) but preserves structural rhythm.
- **Limits (honest)**: generalizes over **15 generator models only** (the AI ceiling; full generality needs a 2nd
  AI-fiction source — not tested); SINGLE paraphraser (Claude; a 2nd paraphraser untested); lexical drop is variable
  across realizations (0.825/0.781/0.919 — r3 barely laundered), so the averaging does real work; modhuman = amateur
  Reddit fiction vs AI fiction (the human pile is not "great prose").
- **GENERALITY (Z2, the honest correction)**: the effect is CONDITIONAL, not general. Tested with an independent
  paraphraser (back-translation EN↔DE) and a 2nd AI source (GPT-4o, Gryphe), both validity-gated (report_z2_*.json):
  - 2nd paraphraser (back-translation, lars1234): diff-in-drops **+0.039, CI [−0.031, 0.111]** — same DIRECTION as F5
    but ns (single deterministic BT realization is underpowered, like the Claude single-realization was). Not a refutation.
  - 2nd source (GPT-4o, back-translation): diff-in-drops **−0.008, CI [−0.085, 0.068]** — the advantage VANISHES.
    The clean A-vs-B comparison (same BT paraphraser, only the source changes: +0.039 → −0.008) shows the effect is
    SOURCE-DEPENDENT. See F6: prosody barely flags GPT-4o at all (0.838 vs 0.925), so there is little robust prosodic
    AI-signal to preserve.
- **Implication (revised, honest)**: prosody's paraphrase-robustness advantage is REAL but NARROW — it holds for
  detecting WEAKER open models with distinctive non-human rhythm, and does NOT generalize to a frontier model (GPT-4o)
  whose prose is already human-like. The general claim "prosody = paraphrase-robust AI detector" is NOT supported. The
  project's payoff is thus a conditional, weak-model effect — honest, publishable as such, but not the clean win the
  multi-realization confirmation alone suggested. Cleanest remaining test: Claude-paraphrased GPT-4o (isolate source
  under the F5 paraphraser) + multi-pivot back-translation realizations.

### F6: Prosody's human-vs-AI signal is much WEAKER for a frontier model (GPT-4o, 0.838) than for weak open models (0.925) — the AI-detection value is a "weak-model" signal (paraphraser-independent)
- **Evidence** (Z2 Part B, report_z2_gpt4o.json): the prosody detector's ORIGINAL (pre-paraphrase) human-vs-AI balanced
  accuracy is **0.838 for GPT-4o** vs **0.925 for lars1234's weak models** and 0.89–0.93 for the F2 modhuman-vs-slop.
  This gap is in the ORIGINAL texts (nothing to do with paraphrase): GPT-4o's prose rhythm is already close to human.
- **Confidence**: medium (single frontier model, story-grouped, n=80 AI). Mechanistically expected (better models →
  more human-like prosody) and consistent with the F5 generality failure (F5 needs a distinctive AI rhythm to be robust to).
- **Implication**: prosody as a human-vs-AI signal degrades as models improve. Its detection value concentrates on
  weaker/older generators; for frontier models the prosodic tell largely disappears. This bounds the practical reach of
  the whole origin-detection story and should be stated plainly in any write-up.

### F7: Speech-grounded prominence does NOT rescue AI-detection (extraction was not the bottleneck there), but gives a BORDERLINE lift on the within-human QUALITY contrast — partial support for the extraction hypothesis, on quality only
- **Method** (values-axis, pre-registered, GPU box, GPU 5): fine-tuned bert-base as a per-word prominence predictor
  on the Helsinki Prosody Corpus (3-level prominence from LibriTTS speech via CWT; 33k sentences, 3 epochs), inferred
  per-word expected prominence on all 320 passages, replaced the dictionary-stress-derived prosody features with
  prominence-derived ones (same feature shapes), and ran head-to-head group-CV vs the old stress features. (src/values_pipeline.py; results/report_values.json.)
- **Evidence** (prom − stress, group-bootstrap CI):
  - AI-DETECTION: modhuman-vs-slop **+0.000 CI[−0.049, 0.045]**; modhuman-vs-gpt4o **−0.019 CI[−0.080, 0.040]**.
    Prominence ≈ dictionary stress — NO improvement. So GPT-4o's human-like prosody (F6) is GENUINE, not an
    extraction artifact; a better front-end does not recover the frontier-model tell.
  - QUALITY: great-vs-modhuman STRESS 0.756 → PROMINENCE **0.825**, **+0.069 CI[0.000, 0.147]** — improves the
    within-human quality contrast; CI lower bound grazes 0 (BORDERLINE, like the F5 near-miss). Prominence (0.825)
    also beats lexical (0.762) here.
- **Confidence**: the AI-detection null is solid (clean, consistent across both contrasts). The quality lift is a
  real but borderline near-miss.
- **Limits**: quality CI grazes 0 (not decisively significant); great = 10 source books (few groups → wide CI, the
  chronic power limit); raw accuracy, not the residualized-against-lexical binding number; the prominence model is
  trained on LibriTTS audiobook (read-aloud) register, which may not fully transfer to our genres.
- **Implication**: the recurring "is it the EXTRACTION?" question gets a SPLIT answer. NO for the origin/detection
  axis — the signal is genuinely weak there and dictionary stress was not the limiter (this closes D14's open
  representation question for detection). A tantalizing PARTIAL YES for within-human QUALITY — a richer per-unit
  signal nudges the project's hardest contrast up by ~0.07. Worth ONE powered follow-up (more great-book groups /
  the residualized binding test) before any final conclusion on the quality axis.

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
- **P6 (the strong-axis pivot, D17 — NOT yet tested):** prosody's robust result is human-vs-AI separation
  (modhuman-vs-slop 0.89, F2), not within-human quality ranking (weak, F2/F3). The high-value hypothesis is
  that prosody is a **paraphrase-robust origin/style signal**: lexical-slop detectors collapse when AI text is
  paraphrased (the n-grams change), but rhythmic structure should survive. Killer test (cheap, runs on existing
  data, needs its own pre-registration): paraphrase both the human and AI piles with an LLM, show lexical-slop
  detection degrades under paraphrase while prosody holds. If true, this is the novel contribution and where
  rhythm has a real edge. Currently a prior, not a finding.
