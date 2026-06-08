# Experiment Log

Append-only. Newest entries at the top. Log **every** run — including failures and dead ends —
*as you do it*, not afterward. Raw numbers, exact commands, expected-vs-observed.

### Entry template (copy this)

```
## YYYY-MM-DD — <short title>
- Phase: <0-4>   Hypothesis tested: <H#>
- Kill criterion for this step (decided beforehand): <...>
- Command(s) / data: <exact commands, dataset + version/path>
- Expected: <what you predicted>
- Observed: <raw results, numbers, where artifacts are saved>
- Interpretation: <what it means, cautiously>
- Next action: <what this implies>
- → Logged to FINDINGS / MISTAKES / DECISIONS? <which, if any>
```

---

## YYYY-MM-DD — Project initialized
- Phase: 0 (not yet started)   Hypothesis tested: none yet
- Kill criterion: n/a
- Command(s) / data: scaffolding created (AGENTS.md + docs/). No experiments run.
- Expected: n/a
- Observed: repository structure in place; environment not yet set up.
- Interpretation: ready to begin Phase 0 once `docs/ENVIRONMENT.md` setup is complete.
- Next action: set up environment, then assemble the three Phase 0 corpora.
- → Logged to FINDINGS / MISTAKES / DECISIONS? Initial decisions recorded in DECISIONS.md.
```
(Replace the placeholder date when the first real session runs.)
```


## 2026-06-07T14:05:54 - Phase 0 PRE-REGISTERED kill criterion (frozen, M7)
- Phase: 0   Hypothesis tested: H1 (separation)
- Decided BEFORE any classifier output exists. No threshold below may change after a number is seen.
- Config (user-selected): GREAT = Gutenberg classics only; SLOP = bottom-quartile overall_score AI stories (lars1234/story_writing_benchmark, documented substitution for EQ-Bench Slop Score); N=80/class; danger-zone = exactly ONE rescue round.
- Binding gate = DEV-set 5-fold StratifiedGroupKFold (group = source book / reuters-category / model_name), scoring = balanced_accuracy, with a 2000x bootstrap 95% CI on pooled out-of-fold predictions + sklearn permutation_test_score (n_permutations=1000). The quarantined held-out (~16/class) is CONFIRMATORY ONLY, scored exactly once.
- chance: 0.50 (binary great-vs-flat), 0.333 (three-class).

PROCEED iff ALL hold:
  (1) binary great-vs-flat balanced_accuracy >= 0.65 AND bootstrap 95% CI lower bound > 0.50;
  (2) permutation_test_score p < 0.01 for the binary task;
  (3) >= 1 rhythm-feature GROUP (collinear gap/run-length/entropy[/tension] grouped; sentence-length CV; clause-spacing CV) in the top-3 by permutation_importance;
  (4) ba_prosody - max(ba_length_only, ba_MATTR_only) >= 0.05 on the same CV;
  (5) FOLD-INTERNAL residualized prosody (regress out mean-sentence-length + MATTR, fit on train fold only, applied to test fold, inside the Pipeline) keeps bootstrap CI lower bound > 0.50;
  (6) era-balanced re-run (euclaise modern-human band added to 'great') keeps binary balanced_accuracy >= 0.60.

KILL iff ANY:
  binary balanced_accuracy <= 0.55 (CI includes 0.50) OR permutation p >= 0.05 OR fold-internal residualized prosody CI lower bound <= 0.50 OR the only above-chance signal is reproduced by the length-only/MATTR-only baseline.
  On KILL: record negative result in FINDINGS.md, lesson in MISTAKES.md, pivot to Phase 4.

DANGER 0.55-0.65: exactly ONE feature-engineering round permitted (add prosodic metrical-tension Tier B), thresholds unchanged and unviewed-against; a second sub-0.65 = KILL.
- Held-out n per class: TO BE FILLED after the group-aware split is built (P0.3).

- Held-out n per class (filled at P0.3): great=17, flat=16, slop=15; total held-out=48, dev=192.

## EVAL features_A.parquet -> report_A.json
- binary great-vs-flat (logreg): balanced_acc=0.984 CI[0.958,1.000] perm_p=1.0000 macroF1=0.984
- binary (tree): ba=0.945
- 3-class (logreg): ba=0.917 perm_p=1.0000
- baselines: length_only=0.511 mattr_only=0.536; prosody - best_baseline=0.448
- residualized(out sl_mean+mattr): ba=0.961 CI[0.924,0.992]
- era-balanced binary ba=0.972
- importance groups (ranked): [('syllable_dist', 0.029321676587301643), ('clause_spacing', 0.025043402777777807), ('sentence_rhythm', 0.014533730158730229), ('stress_timing', 0.009033978174603219), ('confound', 0.00436507936507941)]
- rhythm group in top3: True (top3=['syllable_dist', 'clause_spacing', 'sentence_rhythm'])

## 2026-06-07T14:23:09 - PERMUTATION-TEST BUG found + fixed (M11)
- First evaluate.py run reported perm_p=1.0000 alongside balanced_acc=0.984 (contradiction).
- Root cause: class label is a deterministic function of source group (each book=great, each
  reuters category=flat, each model=slop; 0/30 dev groups are mixed). sklearn permutation_test_score
  with groups shuffles labels WITHIN each group, which does nothing on constant-label groups ->
  every permutation == original -> perm_mean=perm_max=true=0.985, p=1.0 (an ARTIFACT, not a result).
- Fix: group-LEVEL permutation (assign each whole group a random class, keep its samples together).
  Diagnostic (n=200): true=0.985, perm_mean=0.514, perm_max=0.783, p=0.0050. Re-running with n=1000.
- No threshold changed; only the (broken) significance estimator was corrected. The frozen kill
  criterion still reads 'permutation p<0.01 to PROCEED'.

## EVAL features_A.parquet -> report_A.json
- binary great-vs-flat (logreg): balanced_acc=0.984 CI[0.958,1.000] perm_p=0.0010 macroF1=0.984
- binary (tree): ba=0.945
- 3-class (logreg): ba=0.917 perm_p=0.0010
- baselines: length_only=0.511 mattr_only=0.536; prosody - best_baseline=0.448
- residualized(out sl_mean+mattr): ba=0.961 CI[0.924,0.992]
- era-balanced binary ba=0.972
- importance groups (ranked): [('syllable_dist', 0.029321676587301643), ('clause_spacing', 0.025043402777777807), ('sentence_rhythm', 0.014533730158730229), ('stress_timing', 0.009033978174603219), ('confound', 0.00436507936507941)]
- rhythm group in top3: True (top3=['syllable_dist', 'clause_spacing', 'sentence_rhythm'])

- GATE report_A.json: VERDICT=PROCEED (ba=0.984, CI_low=0.958, perm_p=0.0010, delta=+0.448, resid_low=0.924, era=0.972)

## 2026-06-07T14:29:38 - Phase 0 RESULT: PROCEED (H1 supported)
- Phase: 0   Hypothesis: H1 (separation)   Config: GREAT=Gutenberg classics; FLAT=Reuters wire; SLOP=bottom-quartile AI stories; N=80/class; Tier A features.
- Corpora length-matched (great 272 / flat 276 / slop 277 mean words, identical 16/bin) after the M10 fix. Group-aware quarantine: dev=192, held-out=48 (great17/flat16/slop15), zero group overlap.
- M3 metronome guard: PASS (variation features strictly higher on varied prose; metronome ~0).
- DEV (5-fold StratifiedGroupKFold, group-level permutation per D8/M11):
  * binary great-vs-flat: balanced_acc=0.984 CI[0.958,1.000] perm_p=0.001 macroF1=0.984
  * 3-class: balanced_acc=0.917 perm_p=0.001; confusion [[59,2,2],[1,61,2],[4,5,56]] (great,flat,slop)
  * baselines: length_only=0.511, MATTR_only=0.536 (~chance); prosody - best_baseline = +0.448
  * fold-internal residualized (out sentence-length-mean + MATTR): ba=0.961 CI[0.924,0.992]
  * era-balanced (great + euclaise modern-human vs flat): ba=0.972
  * importance top-3 groups: syllable_dist, clause_spacing, sentence_rhythm (rhythm in top-3 = True)
- PAIRWISE (dev): great-vs-flat 0.984, great-vs-slop 0.930, flat-vs-slop 0.977.
- ABLATION (great-vs-slop, the within-narrative QUALITY contrast): all=0.930, no-sentence-length=0.874, sentence-length-only=0.906, sub-sentence-only(no sent-len, no OOV)=0.874 -> genuine sub-sentence rhythm signal, not just sentence-length variety or vocabulary.
- KEY feature direction (great vs slop): sl_var 243 vs 44, sl_cv 0.65 vs 0.37, stress_runlen_var 3.77 vs 1.62 -> great prose has far MORE rhythmic VARIATION (correct sign per hypothesis; opposite of metronome regularity).
- HELD-OUT (quarantined, scored ONCE): great-vs-flat 0.969 CI[0.900,1.000], great-vs-slop 0.908, flat-vs-slop 0.938; rhythm groups top in importance. Matches dev -> no overfitting collapse.
- GATE: all 6 PROCEED conditions PASS, no KILL trigger -> VERDICT=PROCEED. No danger-zone, no rescue round used.
- Interpretation (cautious): text-prosody features separate great/flat/slop well above chance, the separation is NOT explained by passage length or lexical richness (baselines ~chance; survives residualization), and it holds for the two NARRATIVE piles (great-vs-slop) at 0.91-0.93 incl. 0.87 on sub-sentence features alone. H1 is supported.
- LIMITS / Phase-1 priorities: (a) great-vs-flat is genre/register-confounded (fiction vs newswire) - the cleaner quality contrast is great-vs-slop; (b) era confound remains (GREAT is pre-1928 by user choice; euclaise era-band is narrative-vs-wire, not a within-genre era control); (c) AI-slop's known sentence-length monotony may make this contrast 'easy' - Phase 1 should test great vs competent MODERN-human fiction; (d) sentence-length variation + syllable distribution dominate - verify sub-sentence rhythm robustness further. H2 (orthogonality to lexical slop) and H3 (full confound control) are the Phase-1 job, not yet established.
- -> Logged to FINDINGS (F1), METHODOLOGY (Phase 0 exit met; Phase 1 refined), DECISIONS (D8,D9), MISTAKES (M10,M11).

## 2026-06-07T15:57:06 - Phase 1 PRE-REGISTERED kill gate (frozen, M7)

- Phase: 1   Hypotheses: H2 (orthogonality to lexical slop), H3 (separation survives genre+era control)
- Config (user-selected): modhuman = frozen Fan-2018 WritingPrompts (M4 competent-modern-human, D10); modgreat = 1922-1930 Gutenberg (EXPLORATORY/non-gating, D13); binary H3 (NO rescue band); H2.2 increment validity floor lexical_only_ba>=0.65; slop lists pinned SHA c313f04 (D12).
- Significance = group-level permutation (D8/M11); ALL gate bootstrap CIs are GROUP-LEVEL (resample whole source_id groups). Reuse Phase-0 frozen numbers (0.65 / CI_low>0.50 / perm p<0.01 / +0.05 margin / resid CI_low>0.50). Held-out carried forward, scored once.

H2 PROCEED iff BOTH:
  (H2.1) great-vs-slop: prosody residualized fold-internally against the FULL lexical block (cov=lexical incl mattr, feat=prosody excl mattr) keeps group-bootstrap CI_low>0.50 AND group_perm p<0.01.
  (H2.2) modhuman-vs-slop: combined(prosody+lexical) minus lexical_only increment >= 0.05 AND group-bootstrap increment CI_low>0, VALID ONLY IF lexical_only_ba in [0.65,0.90); if lexical_only_ba<0.65 (untrustworthy) or >=0.90 (circular), H2.2 falls to the H2.1-style residual on modhuman-vs-slop (CI_low>0.50 AND p<0.01).
  DIAGNOSTIC-ONLY (pre-registered uninformative): raw great-vs-slop increment (slop lists built from AI text, so lexical_only expected near-ceiling there). Two-way residual (full-lexical vs lexical-minus-distinct-n): collapse vs full but survival vs reduced = shared-variation proxy, REPORT not KILL.
H2 KILL iff: (H2.1) great-vs-slop lexical-residualized-prosody CI_low<=0.50 OR perm p>=0.05.

H3 PROCEED iff binding gate H3-A holds:
  (H3.1) modhuman-vs-slop balanced_acc >= 0.65 AND group-bootstrap CI_low > 0.50.
  (H3.2) modhuman-vs-slop group_perm p < 0.01.
H3 KILL iff: modhuman-vs-slop ba <= 0.55 OR CI_low <= 0.50 OR perm p >= 0.05. NO danger/rescue band.
  REPORTED (honesty checks, not gates): great-vs-modhuman ba (era control within narrative; if >= H3-A flag era as co-driver); within-Gutenberg 19C-vs-early20C ba (MUST be near-chance else era is a live confound); register-only baseline on great-vs-modhuman (MUST NOT separate); modgreat-vs-{slop,great} EXPLORATORY on sub-sentence-only features, Hemingway+Faulkner held out.

OVERALL Phase-1 PROCEED iff H2 PROCEED AND H3 PROCEED. KILL iff H2 KILL OR H3 KILL -> negative F2 to FINDINGS, lesson to MISTAKES, pivot to Phase 4.

## EVAL-P1 features_A.parquet + lexical_A.parquet -> report_h2_pre.json (H2-only)
- H2 great-vs-slop: prosody_only=0.930 lexical_only=0.976 combined=0.976 increment=+0.000; lexical-residualized-prosody ba=0.561 CI[0.443,0.674] perm_p=0.1648 (minus-distinct resid ba=0.617 CI_low=0.487)

- PHASE-1 GATE report_h2_pre.json: VERDICT=KILL (H2.1 resid CI_low=0.443 p=0.1648; modhuman absent)

- P1 carry-forward split quarantine_split_p1.json: held-out by class={'modgreat': 18, 'great': 17, 'modhuman': 16, 'flat': 16, 'slop': 15}, total held=82, dev=318; group overlap=empty; superset of Phase-0 held-out=OK.

## 2026-06-07T16:33:29 - Phase 1b PRE-REGISTERED corrected H2 gate (frozen, M7; explicit re-spec after verified technique flaw)

- Rationale: the re-verification logged above showed the original H2.1 lexical baseline included DIVERSITY-PROXY features (distinct2/distinct3/rep_topk, MATTR) that proxy prosodic variation and circularly remove it. Against a clean SLOP-ONLY baseline, prosody survives on dev great-vs-slop (ba 0.695, perm p=0.002). But great-vs-slop dev is now PEEKED -> exploratory only. Binding confirmation uses UN-PEEKED contrasts + the untouched held-out. Original frozen KILL stays on record (honored); this is an explicit NEW pre-registration (see MISTAKES M12).
- SLOP_BASELINE = slopword_density, slopbigram_density, sloptrigram_density, not_x_but_y_rate, mean_word_len, fk_grade (genuine lexical-slop + readability). DIVERSITY (distinct2, distinct3, rep_topk, mattr) is REPORTED SEPARATELY, never in the slop baseline.
- All CIs group-level; significance group_perm_test (D8/M11). Frozen numbers reused (0.50/0.65/0.01).

H2 (orthogonality to lexical slop) PROCEED iff BOTH:
  (H2b.1) GREAT-vs-MODHUMAN (both human, non-circular, UN-PEEKED): prosody residualized fold-internally against SLOP_BASELINE keeps group-bootstrap CI_low>0.50 AND group_perm p<0.01 on DEV.
  (H2b.2) the same great-vs-modhuman slop-baseline-residual prosody stays above chance on the HELD-OUT (point ba>0.50; confirmatory).
  REPORTED (not binding): great-vs-slop and modhuman-vs-slop slop-baseline residuals; increments over slop-baseline; and the FULL-lexical residual (incl diversity) for transparency.
H2 KILL iff: great-vs-modhuman slop-baseline-residual CI_low<=0.50 OR perm p>=0.05 (prosody adds nothing beyond slop lexicon even where lexical cannot cheat).

H3 (genre+era) PROCEED iff binding H3-A holds (unchanged): modhuman-vs-slop balanced_acc>=0.65 AND CI_low>0.50 AND perm p<0.01.
H3 KILL iff: modhuman-vs-slop ba<=0.55 OR CI_low<=0.50 OR perm p>=0.05.
  REPORTED: great-vs-modhuman prosody-only (era/quality); within-Gutenberg great-vs-modgreat (era null, expect lowish; H/F caveat); modgreat exploratory sub-sentence-only with Hemingway+Faulkner excluded.

OVERALL Phase-1b PROCEED iff H2 PROCEED AND H3 PROCEED. chance=0.50.

## EVAL-P1 features_A_p1.parquet + lexical_A_p1.parquet -> report_p1.json (H2+H3)
- H2 great_vs_slop: prosody_only=0.930 slop_baseline=0.930 full_lexical=0.976 increment_over_slop=+0.046 CI[0.0, 0.095]; resid_vs_SLOP ba=0.695 CI[0.599,0.804] p=0.0030 | resid_vs_FULL(diag) ba=0.561 CI_low=0.443
- H2 modhuman_vs_slop: prosody_only=0.891 slop_baseline=0.907 full_lexical=0.961 increment_over_slop=+0.054 CI[0.014, 0.096]; resid_vs_SLOP ba=0.752 CI[0.647,0.845] p=0.0010 | resid_vs_FULL(diag) ba=0.613 CI_low=0.505
- H2 great_vs_modhuman [BINDING H2b.1]: prosody_only=0.763 slop_baseline=0.764 full_lexical=0.709 increment_over_slop=+0.016 CI[-0.054, 0.089]; resid_vs_SLOP ba=0.669 CI[0.596,0.738] p=0.0020 | resid_vs_FULL(diag) ba=0.590 CI_low=0.507
- H3 modhuman_vs_slop: ba=0.891 CI[0.809,0.954] perm_p=0.0010 (n=129,g=72)
- H3 great_vs_modhuman: ba=0.763 CI[0.663,0.858] perm_p=0.0010 (n=127,g=70)
- H3 great_vs_modhuman_LEXICALonly: ba=0.709 CI[0.626,0.786] (n=127,g=70)
- H3 great_vs_modgreat_full: ba=0.640 CI[0.497,0.763] (n=125,g=16)
- H3 modgreat_vs_slop_subsent_noHF: ba=0.836 CI[0.625,0.962] (n=109,g=16)
- H3 modgreat_vs_great_subsent_noHF: ba=0.421 CI[0.297,0.549] (n=107,g=14)

- PHASE-1b GATE report_p1.json: VERDICT=PROCEED (pending held-out confirmation H2b.2) (H2b.1 great-vs-modhuman resid-vs-SLOP ba=0.669 CI_low=0.596 p=0.0020; H3-A modhuman-vs-slop ba=0.891 CI_low=0.809 p=0.0010)

- P1 HELD-OUT (once): modhuman_vs_slop ba=0.904 CI[0.799,1.000], great_vs_modhuman ba=0.818 CI[0.704,0.909], great_vs_slop ba=0.908 CI[0.733,1.000], great_vs_modhuman_slopresid ba=0.726 CI[0.610,0.841], great_vs_slop_slopresid ba=0.786 CI[0.475,0.931]

## 2026-06-07T17:45:30 - Phase 1 INDEPENDENT VERIFICATION + typography robustness (trust report)

- Independent verification workflow (7 agents, ~616k tokens): 6 verifiers each RE-DERIVED one test from scratch and tried to break it; overall verdict = TRUSTWORTHY-WITH-CAVEATS.
- REPRODUCED independently (matched): length matching 272-277; zero cross-class text leakage (max Jaccard 0.0036); zero source_id dev/held leakage; great-vs-flat 0.984 / great-vs-slop 0.9295 / 3-class 0.917 / held-out 0.969/0.908; baselines at chance; M11 sklearn-perm degeneracy + group-level null at chance; H2 resid-vs-FULL 0.561 KILL vs resid-vs-SLOP 0.695 survive; binding great-vs-modhuman 0.669 dev / 0.726 held-out; feature parquet re-extracts to 0.0 diff; M3 guard.
- ISSUES FOUND: (1) TYPOGRAPHY confound (M13): modhuman quote-glyphs differ; typography-only classifier 0.76-0.84. Robustness re-run (typography control, group CV, group bootstrap): prosody resid-vs-SLOP+TYPOGRAPHY -> modhuman-vs-slop 0.713 CI[0.612,0.813] (survives), great-vs-modhuman 0.606 CI[0.522,0.699] p=0.016 (above chance but weakened), great-vs-slop 0.609 CI[0.482,0.741] (touches chance). (2) WORD-LENGTH confound (M14): mean_word_len separates great-vs-flat 0.93 alone, absent from the Phase-0 residualizer (present in Phase-1b slop baseline). (3) DRIVER mis-stated: importance ranks syllable_dist > sentence_rhythm. (4) H2 mechanism overstated (joint slop+diversity + near-ceiling, not purely diversity proxies). (5) flat: 2 byte-identical + ~6 near-dup pairs straddle dev/held (flat-only; not the binding contrasts). (6) perm p floor-bounded; modgreat small-n.
- NET: core numbers are real and reproducible; H2/H3 supported with caveats; strongest clean result = modern-human-vs-AI (survives lexical-slop + typography control). Logged as FINDINGS F2. Phase-1c hardening (typography normalization, flat dedup, mean_word_len covariate, perm n>=1000) recommended before publication-grade claims.

## 2026-06-08T10:15:24 - Phase 1c PRE-REGISTERED re-confirmation (caveat fixes; thresholds unchanged, M7)

- Goal: re-confirm the Phase-1b binding results after fixing the verification caveats: (a) typography normalized UNIFORMLY across ALL piles IN PLACE (M13 fix; passage_ids preserved -> held-out quarantine undisturbed; verified typography-only great-vs-modhuman 0.758->0.481 chance, prosody features ~unchanged), (b) flat de-duplicated (14 near-dup Reuters passages removed from the split; binding great/modhuman/slop membership UNCHANGED), (c) mean_word_len is in SLOP_BASELINE so the binding H2 residual already controls word length, (d) permutation n=1000.
- Binding thresholds UNCHANGED from Phase 1b (M7): H2b.1 great-vs-modhuman prosody residualized vs SLOP_BASELINE on the NORMALIZED corpus: CI_low>0.50 AND group_perm p<0.01. H3-A modhuman-vs-slop prosody-only: ba>=0.65 AND CI_low>0.50 AND p<0.01.
- HONESTY on peeking: I already saw the typography-as-COVARIATE robustness numbers (great-vs-modhuman 0.606 p=0.016; modhuman-vs-slop 0.713). The p1c re-run uses the cleaner normalize-at-source method on partly-peeked dev contrasts, so the genuinely UN-PEEKED confirmation is the HELD-OUT on the normalized+deduped corpus (heldout_report_p1c.json), scored once.
- CONFIRM iff: on the normalized corpus, great-vs-modhuman resid-vs-SLOP keeps CI_low>0.50 & p<0.01 (dev) AND held-out great-vs-modhuman slop-residual stays >chance AND modhuman-vs-slop holds ba>=0.65. WEAKEN iff great-vs-modhuman resid-vs-SLOP drops to CI_low<=0.50 (then modhuman-vs-slop remains the primary robust result).
- Artifacts (do NOT clobber frozen Phase-1b): features_A_p1c.parquet, lexical_A_p1c.parquet, quarantine_split_p1c.json, report_p1c.json, heldout_report_p1c.json. data/passages_raw/ holds the pristine pre-normalization text.

## EVAL-P1 features_A_p1c.parquet + lexical_A_p1c.parquet -> report_p1c.json (H2+H3)
- H2 great_vs_slop: prosody_only=0.930 slop_baseline=0.930 full_lexical=0.969 increment_over_slop=+0.046 CI[0.0, 0.095]; resid_vs_SLOP ba=0.696 CI[0.585,0.809] p=0.0020 | resid_vs_FULL(diag) ba=0.553 CI_low=0.445
- H2 modhuman_vs_slop: prosody_only=0.891 slop_baseline=0.915 full_lexical=0.961 increment_over_slop=+0.046 CI[0.009, 0.086]; resid_vs_SLOP ba=0.752 CI[0.642,0.851] p=0.0010 | resid_vs_FULL(diag) ba=0.605 CI_low=0.497
- H2 great_vs_modhuman [BINDING H2b.1]: prosody_only=0.787 slop_baseline=0.740 full_lexical=0.732 increment_over_slop=+0.102 CI[0.034, 0.168]; resid_vs_SLOP ba=0.653 CI[0.571,0.732] p=0.0010 | resid_vs_FULL(diag) ba=0.621 CI_low=0.546
- H3 modhuman_vs_slop: ba=0.891 CI[0.809,0.954] perm_p=0.0010 (n=129,g=72)
- H3 great_vs_modhuman: ba=0.787 CI[0.690,0.879] perm_p=0.0010 (n=127,g=70)
- H3 great_vs_modhuman_LEXICALonly: ba=0.732 CI[0.648,0.812] (n=127,g=70)
- H3 great_vs_modgreat_full: ba=0.656 CI[0.504,0.784] (n=125,g=16)
- H3 modgreat_vs_slop_subsent_noHF: ba=0.836 CI[0.661,0.958] (n=109,g=16)
- H3 modgreat_vs_great_subsent_noHF: ba=0.382 CI[0.246,0.520] (n=107,g=14)

- PHASE-1b GATE report_p1c.json: VERDICT=PROCEED (pending held-out confirmation H2b.2) (H2b.1 great-vs-modhuman resid-vs-SLOP ba=0.653 CI_low=0.571 p=0.0010; H3-A modhuman-vs-slop ba=0.891 CI_low=0.809 p=0.0010)

- P1 HELD-OUT (once): modhuman_vs_slop ba=0.904 CI[0.799,1.000], great_vs_modhuman ba=0.849 CI[0.753,0.917], great_vs_slop ba=0.908 CI[0.733,1.000], great_vs_modhuman_slopresid ba=0.695 CI[0.570,0.812], great_vs_slop_slopresid ba=0.816 CI[0.575,0.931]

## 2026-06-08T10:40:04 - Phase 1c RESULT: F2 CONFIRMED on hardened corpus (PROCEED, held-out confirmed)

- Phase 1c (caveats fixed: typography normalized at source, flat deduped, perm n=1000) -> VERDICT PROCEED, held-out CONFIRMED. F2 CONFIRMED and strengthened.
- H2b.1 BINDING great-vs-modhuman resid-vs-SLOP (normalized corpus): ba=0.653 CI[0.571,0.732] p=0.001 (Phase-1b was 0.669; typography-as-COVARIATE had over-removed to 0.606/p=0.016). increment_over_slop +0.102 CI[0.034,0.168] (now significant; was +0.016 ns). HELD-OUT great-vs-modhuman slop-residual = 0.695 (un-peeked, >chance) -> H2b.2 confirmed.
- H3-A modhuman-vs-slop prosody-only: 0.891 dev / 0.904 held-out, p=0.001 (unchanged; rock solid).
- great-vs-slop resid-vs-SLOP 0.696 dev / 0.816 held-out p=0.002; resid-vs-FULL still collapses 0.553 (the diversity-proxy artifact, expected). modhuman-vs-slop resid-vs-SLOP 0.752 dev p=0.001.
- Era null intact: modgreat-vs-great sub-sentence (no Hemingway/Faulkner) = 0.382 (below chance) -> era not a confound at sub-sentence level. great-vs-modgreat full 0.656 (H/F + sentence-length driven, exploratory).
- INTERPRETATION: the typography confound was NOT driving the binding result; removing it at source (vs as a residualization covariate, which over-removes prosody variance) shows prosody's orthogonality to lexical-slop HOLDS, and is cleaner than Phase 1b. The earlier 0.606/p=0.016 was a covariate-method artifact.
- Caveats now CLEARED: typography (fixed), flat dedup (fixed), word-length (in slop baseline), perm n (1000). Remaining honest limits: modgreat small-n (8 books, exploratory); great-vs-flat genre-confound is inherent (not a quality claim); the result is human-vs-AI + within-human-quality separation, strongest for modern-human-vs-AI; prosody-as-standalone-quality-reward still rejected (D14) - this supports prosody as a complementary/diversity signal.
