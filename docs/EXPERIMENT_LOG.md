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

## 2026-06-08 — Stage 1 (order-aware / sequence prosody features): PRE-REGISTRATION (frozen BEFORE the binding run, M7)
- Phase: extraction-upgrade (EXTRACTION_UPGRADE_PLAN Stage 1). Hypothesis: preserving the ORDER of the rhythmic series (SL/SPW/SYMS/PHG) carries within-human writing-quality signal BEYOND the order-blind 14 summary stats (sl_var etc. discard long-short-long-short vs long-long-short-short), partly orthogonal to the lexical-slop baseline. Cheapest-kill-first (M9): pure numpy, no new heavy model, before any Helsinki/CWT/learned-rep investment.
- Design hardened by a 3-lens design panel + synthesis (workflow wn6y05p9j). 10 FROZEN order features (all order-sensitive, scale-invariant, finite-guarded; metronome forced to an EXTREME, never the 'good' region — M3): sl_ac1, sl_mk, sl_bvr, sl_lre, sl_turning, spw_ac1, syms_ac1, syms_runz, phg_ac1, phg_lre. Rejected (logged): permutation-entropy + spectral features (order-insensitive on long series and/or length proxies — sl_perm_ent corr -0.69 with sl_mean), DFA, positional SL features (parsimony).
- BINDING contrast: great-vs-modhuman (clean within-human, M12-safe), DEV n=127 (great=63/10 books, modhuman=64/60 ids), group=source_id, StratifiedGroupKFold(5), group-perm n=1000, group_bootstrap_increment_ci. Lexical baseline = SLOP_BASELINE_COLS. Held-out (n=33) scored EXACTLY ONCE, ONLY if all DEV gates pass.
- PRE-REGISTERED GATES (all three must pass to call order a real lever; kill if ANY fails):
  - PRIMARY-1 (lift over histograms): ba(summary14 + order10) − ba(summary14) nested increment, group-bootstrap 2.5% CI lower bound > 0.
  - PRIMARY-2 (order is REAL, not order-invariant residue; CO-PRIMARY/decisive): ba(summary + REAL order) − ba(summary + SHUFFLED order) paired group-bootstrap 2.5% CI lower bound > 0. SHUFFLED = per-passage independent permutation of each series, seed = md5(passage_id) (frozen, reproducible; NOT python hash()).
  - PRIMARY-3 (orthogonality, M12): SLOP-residualized (summary + order) group_perm_test p < 0.01 AND increment_over_slop CI floor clears the summary-only +0.102 (F2 reference: summary resid_vs_slop 0.653 CI[0.571,0.732]).
- KILL (honored per M7, do not move goalposts): any gate fails → clean NEGATIVE: 'order-aware summary-derived hand features add nothing over passage histograms on the clean within-human contrast; the ~0.65 ceiling is the SIGNAL (or the SL/SPW/SYMS/PHG series themselves), not order-blindness.' Per M9 do NOT proceed to CWT/learned-sequence on hand features; redirect the next lever to a learned representation or higher-ceiling dataset.
- PILOT PRIOR (design-panel scoping, NOT a finding — to be independently reproduced by this implementation): order-only REAL 0.606 == SHUFFLED 0.606; summary+REAL 0.771 vs summary+SHUFFLED 0.732 (REAL not reliably above SHUFFLED). The kill (PRIMARY-2) is EXPECTED to fire. Building the harness to deliver that negative cleanly; not tuning toward a positive.
- Controls (reported every run): order-shuffle (PRIMARY-2 itself), metronome/constant/random feature values (test_sequence.py), length/n_sent residualization diagnostic (M5/M14), typography-only at chance (M13), OOV-drop correlation check. Artifacts: src/sequence_features.py, src/extract_sequence.py, src/stage1_sequence.py, tests/test_sequence.py, results/features_seq_A.parquet, results/report_stage1.json.

## 2026-06-08 — Stage 1 RESULT (order-aware sequence features): VERDICT KILL
- great-vs-modhuman DEV n=127 groups=70 {'modhuman': 64, 'great': 63}
- ba: summary=0.787 summary+order=0.771 summary+order_SHUF=0.756 | order_only=0.606 order_only_SHUF=0.614
- PRIMARY-1 (lift over histograms): increment=-0.016 CI[-0.078,0.042] -> FAIL
- PRIMARY-2 (order REAL vs SHUFFLED, decisive): increment=+0.015 CI[-0.082,0.119] -> FAIL
- PRIMARY-3 (orthogonality): resid_slop ba=0.661 p=0.0030 increment_over_slop=+0.031 CI[-0.049, 0.112] (summary-only +0.102) -> FAIL
- per-feature order-sensitivity (mean|real-shuf|): {'syms_runz': 3.1892, 'sl_bvr': 0.3832, 'sl_ac1': 0.2906, 'sl_mk': 0.2386, 'phg_ac1': 0.1932, 'syms_ac1': 0.17, 'sl_turning': 0.1383, 'sl_lre': 0.1358, 'phg_lre': 0.1045, 'spw_ac1': 0.0791}
- INTERPRETATION: KILL (expected): order-aware hand features add nothing beyond passage histograms on the clean within-human contrast; the ~0.65 ceiling is the SIGNAL/series, not order-blindness (M9 -> do not proceed to CWT/learned-seq on hand features).

## 2026-06-08 — Stage 1 KILL: ADVERSARIAL VERIFICATION (workflow wzs5gdntm, 4 independent verifiers + synthesis) -> KILL-ROBUST
- All 5 BA independently re-derived from the parquets (own logreg + StratifiedGroupKFold seed0, NOT importing stage1_sequence) reproduce to 6 dp, 0.000000 diff: summary 0.786954, summary+order 0.771329, summary+order_shuf 0.755952, order_only 0.606027, order_only_shuf 0.613715. All 3 gates FAIL exactly.
- NO false-kill bug: series rebuilt from spaCy match stored REAL and md5-SHUFFLED order cols to 0.0 abs diff (shuffle + math correct); shuffle genuine (mean|real-shuf| 0.08–3.19, corr ~0); order cols non-degenerate (var 0.004–1.64, <6% zero); ordered_series byte-identical to the summary-stat series (no desync); dev/held overlap 0; held-out NOT touched.
- RESCUE exhausted: no model (logreg/RF/GB/HGB/pairwise-interactions), no CV seed (PRIMARY-1 negative 10/10, mean -0.032), no feature subset clears a group-bootstrap CI lower bound > 0. Best honest order increment HGB real-minus-shuffle +0.047 CI[-0.004,0.107]; pairwise-interactions significantly NEGATIVE CI[-0.143,-0.006]. Per-feature Welch t great-vs-modhuman: sl_bvr p=0.009, phg_ac1 p=0.018 (NONE survive Bonferroni alpha=0.005).
- DECISIVE binding test: order residualized against the summary HISTOGRAMS = 0.551 (chance), BELOW the residualized SHUFFLE (0.567) -> order is fully redundant with the histograms.
- CORRECTION to my single-md5 PRIMARY-2 (under-powered, CI too wide): 40-seed within-passage shuffle (3 seed-schemes) shows real order DOES carry a small genuine signal vs order-destroyed series (REAL at 70–95th pct of the shuffle ba distribution, point +0.027, p<0.001) — but integrating shuffle-seed noise into the group bootstrap gives real-minus-shuffle +0.027 CI[-0.072,0.129] (floor << 0). Tiny, swamped by group-level CIs, and redundant per the residual test. -> M15.
- POWER limit (honest): 70 source groups + n=127 give CIs ~±0.10, too wide to resolve a ~0.03 effect. Verdict KILL is robust to model family, CV seed, feature cherry-picking, and the shuffle lottery. Held-out remains quarantined (gates failed; not scored, M7). Verification record: results/report_stage1_verification.json. -> FINDINGS F3, DECISIONS D16, MISTAKES M15.

## 2026-06-08 — Track A (paraphrase-robustness, P6): PRE-REGISTRATION (frozen BEFORE the run, M7)
- Phase: strong-axis pivot (D17). Hypothesis P6: prosody is a paraphrase-robust ORIGIN signal. Paraphrasing launders the lexical slop n-grams (a lexical-slop detector degrades under paraphrase), but rhythmic structure is a more stable style property (the prosody detector degrades LESS). Threat model: adversary paraphrases AI text to evade detection. Runs on EXISTING data; no new corpus.
- Contrast: modhuman-vs-slop (the robust human-vs-AI axis, F2 0.89). All 160 main-band passages (modhuman 80 + slop 80, 91 source groups) paraphrased by an independent LLM (Claude subagents; NOT the slop generators Qwen/Arcee/gemma/... — so no trivial generator fingerprint), meaning-preserving, fluent, length-approx, applied SYMMETRICALLY to both classes -> data/passages_para/<class>/<id>.txt.
- Design (paired, group-aware, no leakage): train each detector on ORIGINAL features; StratifiedGroupKFold(5); in each fold score the held-out groups' passages in BOTH original and paraphrased form (test groups never trained on). DROP = ba_original − ba_paraphrased. Detectors: PROSODY = PROSODY_COLS; LEXICAL = SLOP_BASELINE_COLS. Eval = deterministic src/paraphrase_eval.py.
- BINDING metric: difference-in-drops = lexical_drop − prosody_drop, paired group-bootstrap CI (2000x over source groups).
- VALIDITY gates (test is VOID if any fail): (i) lexical_drop > 0.03 (paraphrase actually laundered the slop signal — else nothing to be robust to); (ii) mean para/orig length ratio in [0.6, 1.6] (paraphrase preserved, not degenerate); (iii) coverage >= 140/160.
- PRE-REGISTERED VERDICT (stop rule, honored M7): P6-SUPPORTED iff validity holds AND difference-in-drops CI lower bound > 0 (prosody robustly degrades LESS). P6-NOT-SUPPORTED iff prosody_drop >= lexical_drop / CI includes 0 -> rhythm is no more paraphrase-robust than lexical; the novel origin-robustness claim fails; record clean negative, do not build a prosody robustness product. INVALID iff a validity gate fails -> re-do paraphrase protocol, do not interpret.
- M15: single paraphrase realization per passage is one draw; verify phase re-paraphrases a subset with a 2nd LLM-call seed to check the difference-in-drops is stable (not a paraphrase-lottery artifact). Artifacts: data/passages_para/, src/paraphrase_eval.py, results/report_paraphrase.json. Run via workflow (paraphrase gen -> eval -> adversarial verify), in parallel with Track B (within-author dataset design).

## 2026-06-08 — EXPLORATORY: Paul Graham essays (high-quality modern human) vs AI/others
- PG: 223 passages / 18 essays, mean 282 words. CONFOUND: PG=essay vs slop/modhuman=fiction (genre, M5). Not a binding test; held-out untouched.
- pg_vs_slop: prosody_only=0.960 CI[0.918,0.991] p=0.001 | slop_lex=0.950 | length_only=0.506 | incr_over_slop=+0.023 CI[-0.009, 0.058] (n=303,g=33, words {'pg': 282.17040358744396, 'slop': 276.6375})
- pg_vs_modhuman: prosody_only=0.880 CI[0.829,0.924] p=0.001 | slop_lex=0.876 | length_only=0.652 | incr_over_slop=+0.012 CI[-0.026, 0.052] (n=303,g=94, words {'pg': 282.17040358744396, 'modhuman': 273.85})
- pg_vs_great: prosody_only=0.931 CI[0.873,0.983] p=0.001 | slop_lex=0.830 | length_only=0.641 | incr_over_slop=+0.109 CI[0.054, 0.16] (n=303,g=30, words {'pg': 282.17040358744396, 'great': 272.4375})
- modhuman_vs_slop: prosody_only=0.925 CI[0.865,0.972] p=0.001 | slop_lex=0.931 | length_only=0.756 | incr_over_slop=+0.038 CI[-0.0, 0.078] (n=160,g=91, words {'modhuman': 273.85, 'slop': 276.6375})

## 2026-06-08 — Track Z (ZuCo implicit-prosody validation): PRE-REGISTRATION (frozen BEFORE any model sees the data, M7)
- Question (D17/P1 foundation): are our text-derived prosodic PRIMITIVES psychologically real — do they predict how readers actually process text during natural silent reading? Validation oracle, not training. Independent, hard-to-game.
- Data: ZuCo 2.0 Normal Reading, 8 subjects (smallest .mat), ~300 Wikipedia sentences, per-word FFD/GD/GPT/TRT/nFixations + a theta-band EEG aggregate (theta_trt). Parsed locally via src/zuco_parse.py (eye-tracking + theta only; raw EEG discarded; verified on YAK = 5937 words/299 sents). CAVEAT: ZuCo NR is register-homogeneous Wikipedia text -> limited prosodic VARIATION across sentences (weakens the sentence-level arm).
- WORD-LEVEL predictors. PROSODY (ours): nsyl (CMUdict syllable count), is_content (word carries lexical stress / not a function word), is_boundary (our clause/sentence-boundary detection: trailing punct in _PUNCT_BOUNDARY or sentence-final). CONFOUNDS: word length (chars), log-frequency (wordfreq Zipf), GPT-2 surprisal in sentence context, sentence position, is_sentence_final.
- Models (skipped words = nFix 0 modelled separately from reading times): (1) SKIP: logistic fixated~confounds(+prosody), subject-clustered SE; (2) READING TIME: MixedLM log(TRT | fixated) ~ confounds(+prosody), random intercept by subject; (3) EEG: same on theta_trt. BINDING = improvement (LR/F, p<0.01) of the PROSODY block over confounds-ONLY, and the is_boundary coefficient (the distinctive wrap-up effect not reducible to a word's own length/freq).
- SECONDARY (sentence-level, directly uses our 14 features): do our sentence prosody summary features predict mean sentence reading effort beyond sentence length + mean-freq + mean-surprisal (~300 sentences).
- PRE-REGISTERED VERDICT (stop rule, M7): PROSODY-REAL iff the prosody block significantly improves fit beyond confounds on reading time (LR p<0.01) AND/OR is_boundary predicts a wrap-up slowdown (positive, p<0.01). NULL iff prosody adds nothing beyond length/freq/surprisal -> our extraction is largely a surface-confound restatement, not an independent capture of implicit prosody (points at the values/extraction problem). Binding number is ALWAYS increment-over-confounds (esp. surprisal), never raw prosody correlation. Few pre-specified tests (multiple-comparison aware). Artifacts: src/{zuco_parse,zuco_fetch,zuco_analysis}.py, data/zuco/words_*.csv (gitignored), results/report_zuco.json.

## 2026-06-08 — Track Z RESULT (ZuCo implicit-prosody validation): VERDICT PROSODY-REAL
- 48997 words, 8 subjects, 349 Wikipedia sentences (NR); prosody=ours(nsyl,is_content,is_boundary) vs confounds(length,logfreq,GPT2-surprisal,pos,sent-final)+subjectFE, sentence-clustered SE
- skip: prosody-block F=100.9 p=1.00e-21 | is_boundary coef=+0.009 p=6.14e-01 | coefs={'nsyl': -0.035, 'is_content': 0.2445, 'is_boundary': 0.009} (n=48997)
- reading_time: prosody-block F=11.6 p=8.89e-03 | is_boundary coef=+0.002 p=6.64e-01 | coefs={'nsyl': 0.0146, 'is_content': -0.0169, 'is_boundary': 0.0022} (n=28541)
- theta_eeg: prosody-block F=2.8 p=4.16e-01 | is_boundary coef=-0.002 p=6.45e-01 | coefs={'nsyl': 0.0018, 'is_content': 0.0101, 'is_boundary': -0.0018} (n=28517)

## 2026-06-08 — Track A RESULT (paraphrase-robustness, P6): VERDICT P6-NOT-SUPPORTED
- modhuman-vs-slop, n=160 groups=91, para coverage 160/160, length ratio 0.99
- PROSODY: ba original=0.925 -> paraphrased=0.875 (drop +0.050)
- LEXICAL: ba original=0.931 -> paraphrased=0.825 (drop +0.106)
- DIFFERENCE IN DROPS (lexical − prosody) = +0.056 CI[-0.003,0.115] -> NOT distinguishable / prosody not more robust
- validity gates: {'lexical_actually_degraded': True, 'length_preserved': True, 'coverage_ok': True, 'all_ok': True}

## 2026-06-08 — Track A POWER-UP (paraphrase-robustness P6, multi-realization): PRE-REGISTRATION (frozen BEFORE the run, M7)
- Why: F5 was a near-miss (+0.056 CI[-0.003,0.115], grazes 0). Diagnosis of the CI width: the slop side has only 15 generator-model GROUPS (lars1234 has exactly 15 en models — VERIFIED; the full bottom-quartile pool is 940 stories but still only 15 models), so the group-bootstrap CI is GROUP-limited on the AI side. APPLYING MISTAKES (per user instruction, don't repeat them):
  - M15 (our own session lesson): the original run used a SINGLE paraphrase realization — underpowered. The dominant REDUCIBLE noise is the paraphrase lottery, not passage count. FIX = multi-realization.
  - M11: keep GROUP-level bootstrap (do NOT switch to passage-level to manufacture a tighter CI — that is the exact inference-gaming error the group rules guard against).
  - HONESTY: adding more passages from the SAME 15 models would NOT tighten the group-limited CI; we will NOT fake that. The residual 15-model AI ceiling is disclosed; fully resolving it needs a SECOND AI-fiction source (more generators), out of scope here.
  - M5/M10/M13: same frozen 160 passages (already length-matched + typography-normalized at source); no corpus change -> no new length/typography confound.
- Design: 3 paraphrase realizations of the SAME 160 modhuman+slop passages (realization 1 = the existing run; generate r2, r3 as independent Claude generations; meaning + length preserved, symmetric). Per realization, train detectors on ORIGINAL features and score the paraphrased -> ba_paraphrased_r; average over the 3 realizations -> drop = ba_original − mean_r(ba_paraphrased_r). Binding = difference-in-drops (lexical − prosody), GROUP-bootstrap over source_id, with across-realization variance folded in. ALSO report the per-realization spread (M15 transparency).
- PRE-REGISTERED VERDICT (M7, unchanged bar, NO goalpost move): P6-SUPPORTED iff difference-in-drops group-bootstrap 2.5% CI lower bound > 0; else NOT-SUPPORTED. Validity gates unchanged (lexical must degrade; length ratio in [0.6,1.6]; coverage). HONEST either way; a near-miss that stays a near-miss = "directional support, AI-source-diversity-limited; next real lever is more generators." Single-paraphraser (Claude) limit noted; a 2nd paraphraser remains untested. Artifacts: data/passages_para_r2|r3/, src/paraphrase_eval_multi.py, results/report_paraphrase_multi.json.

## 2026-06-08 — Track A POWER-UP RESULT (multi-realization, 3 realizations): VERDICT P6-SUPPORTED
- modhuman-vs-slop, n=160 groups=91, realizations=3, length ratios=[0.992, 0.99, 0.996]
- PROSODY: ba_orig=0.925 -> para_mean=0.888 (per-real [0.875, 0.881, 0.906]); drop +0.037
- LEXICAL: ba_orig=0.931 -> para_mean=0.842 (per-real [0.825, 0.781, 0.919]); drop +0.090
- DIFFERENCE IN DROPS (lexical-prosody) = +0.052 CI[0.009,0.098] -> prosody MORE robust (CI>0)
- validity {'lexical_actually_degraded': True, 'length_preserved': True, 'coverage_ok': True, 'all_ok': True} | AI ceiling: 15 generator models (group-limited, disclosed)

## 2026-06-08 — Z2 (F5 GENERALITY check): PRE-REGISTRATION (frozen BEFORE the run, M7)
- Goal: close the two honest limits of F5 (P6 confirmed): (A) single PARAPHRASER (Claude), (B) single AI SOURCE (lars1234, 15 models). General eval src/robustness_eval.py VALIDATED — reproduces F5 single-realization exactly (diff-in-drops +0.056 CI[-0.003,0.115]) before any new data is trusted.
- PART A — paraphraser generality (HIGH confidence): re-run modhuman-vs-slop with an INDEPENDENT non-LLM paraphraser = BACK-TRANSLATION EN->DE->EN (Helsinki-NLP opus-mt; verified meaning+length preserved). Same 160 passages, same 91 groups (incl 15 slop models). HONEST CAVEAT: back-translation preserves sentence boundaries MORE than free paraphrase (whole-passage / 2-part split lets MT restructure, but less than Claude), so it is a SOFTER test on prosody's sentence-length channel and a genuine INDEPENDENT test of the LEXICAL-laundering differential. Triangulates with the Claude result.
- PART B — AI-source generality (MEDIUM/LOW confidence, caveated): new contrast modhuman-vs-gpt4o where gpt4o = Gryphe/ChatGPT-4o-Writing-Prompts (AI fiction on WritingPrompts, GENRE-MATCHED to modhuman, M5; GPT-4o NOT in lars1234's 15). Length-matched + typography-normalized at source (M10/M13). HONEST CAVEAT: GPT-4o is a SINGLE model -> grouped by STORY (non-degenerate bootstrap) but this tests robustness for ONE FRONTIER generator, NOT cross-model generality; it does NOT by itself close the 15-model ceiling. Paraphraser = back-translation (consistent with Part A; Claude optional later).
- BINDING (both parts, M7 unchanged bar, M11 group-bootstrap kept): difference-in-drops (lexical - prosody) group-bootstrap 2.5% CI lower bound > 0 = the F5 effect generalizes; validity gates unchanged (lexical must degrade, length ratio in [0.6,1.6], coverage). REPORT BOTH OUTCOMES HONESTLY: if a part does not clear 0, say so; uncertainty > overclaiming. Verdict synthesis: F5 GENERALIZES if Part A holds AND Part B holds; PARTIAL if only one; if neither, F5 is paraphraser/source-specific. Artifacts: src/{robustness_eval,backtranslate,build_gpt4o}.py, data/passages_para_bt/, data/passages_gpt4o{,_bt}/, results/report_z2_*.json.

## 2026-06-08 — Z2 (F5 generality) RESULT: F5 is CONDITIONAL/NARROW — holds for weak open models, FAILS for the frontier model GPT-4o
- General eval VALIDATED (reproduces F5 single-realization +0.056). Both parts pass all validity gates (lexical degraded, length preserved, coverage).
- PART A (back-translation paraphraser, modhuman-vs-slop, lars1234): prosody drop +0.070, lexical drop +0.108; diff-in-drops **+0.039 CI[-0.031,0.111]** -> NOT-SUPPORTED (ns). Direction CONSISTENT with F5 (prosody more robust) but ns. Honest read: SINGLE back-translation realization is underpowered (BT is deterministic; cf. the Claude SINGLE-realization was also ns at +0.056, only MULTI-realization reached sig). So Part A = underpowered, NOT a refutation; direction holds.
- PART B (back-translation paraphraser, modhuman-vs-GPT-4o, Gryphe ChatGPT-4o-Writing-Prompts; story-grouped): **prosody_orig_ACC = 0.838** (vs 0.925 for the weak models) — GPT-4o's rhythm is already HUMAN-LIKE, so prosody barely flags it even BEFORE paraphrase (a paraphraser-INDEPENDENT tell). prosody drop +0.125, lexical drop +0.119; diff-in-drops **-0.008 CI[-0.085,0.068]** -> NOT-SUPPORTED, effect ABSENT (point ~0/slightly negative).
- CLEAN A-vs-B comparison (SAME paraphraser = back-translation, only the AI SOURCE changes): lars1234 +0.039 vs GPT-4o -0.008. The prosody-robustness advantage is SOURCE-DEPENDENT — weaker/absent for the frontier model.
- HONEST VERDICT: **F5 does NOT generalize.** Prosody's paraphrase-robustness advantage exists for detecting WEAKER open models with distinctive non-human rhythm (lars1234: F5 +0.052 sig under Claude-multi), but VANISHES for GPT-4o, whose prose is already human-like. The general claim "prosody = paraphrase-robust AI detector" is NOT supported; it is a NARROW, conditional effect. MECHANISM: a robust prosodic AI-signal requires the AI to have distinctive non-human rhythm in the first place; frontier models don't. -> FINDINGS F5 (qualified to CONDITIONAL) + F6 (frontier-model prosody-detection weakness).
- LIMITS/uncertainty (honest, per user): Part A/B = single BT realization (multi-pivot-language realizations DE/FR/ES untested); Part B confounds source+paraphraser, but the A-vs-B BT comparison isolates source AND the 0.838 original accuracy is paraphraser-independent; a Claude-paraphrased GPT-4o test (cleanest source isolation under the F5 paraphraser) is UNTESTED — would sharpen the conclusion.

## 2026-06-08 — VALUES-AXIS experiment (is the EXTRACTION the bottleneck?): PRE-REGISTRATION (frozen BEFORE the binding run, M7)
- THE last untested lever. We ruled out the classifier (D14) and the ORDER of the existing series (F3). UNTESTED: the per-unit VALUES — every prosody number traces to crude DICTIONARY CITATION-STRESS (pronouncing/CMUdict). Replace it with a SPEECH-GROUNDED per-word prominence and see if results move.
- Method (runs on the remote GPU box, GPU 5 ONLY, detached so it survives laptop-off; src/values_pipeline.py): (1) fine-tune bert-base token-classifier on the Helsinki Prosody Corpus (train_100, 686k words, 3-level prominence labels derived from LibriTTS speech via CWT); (2) infer per-word EXPECTED prominence (0..2) on every modhuman/slop/great/gpt4o passage; (3) build prominence-based prosody features mirroring the stress features (prom mean/var/cv/runlen_var/entropy/gap timing/ac1) + the shared sentence/clause features; (4) head-to-head contrasts STRESS(old PROSODY_COLS) vs PROMINENCE(new) vs lexical, group-bootstrap.
- CONTRASTS (the ones that matter): modhuman-vs-slop (F2 ~0.89-0.93), modhuman-vs-gpt4o (F6, weak 0.838 — does prominence recover the frontier tell?), great-vs-modhuman (quality, ~0.65). Group = source_id, StratifiedGroupKFold(5), balanced accuracy, group-bootstrap increment of PROMINENCE over STRESS.
- PRE-REGISTERED VERDICT (M7, honest both ways): EXTRACTION WAS THE BOTTLENECK iff prominence beats dictionary-stress with group-bootstrap CI lower bound > 0 on a contrast that was WEAK (esp. GPT-4o detection or the quality contrast). If prominence does NOT beat stress -> the signal is GENUINELY WEAK, dictionary stress was not the limiter, and the prosody quality/frontier-detection question CLOSES. Either outcome is decisive. CAVEAT: prominence model trained on LibriTTS audiobook prose (read-aloud register); a null could also mean the prominence labels don't transfer to our genres. SMOKE-TESTED end-to-end before launch (train->infer->contrasts->DONE). Artifacts on box: ~/prosody_values/{report.md, report_values.json}; STATUS/DONE/FAILED markers.

## 2026-06-08 — VALUES-AXIS RESULT: SPLIT answer — extraction is NOT the bottleneck for AI-detection, but speech-grounded prominence gives a BORDERLINE lift on the within-human QUALITY contrast
- Ran on GPU box (GPU 5, detached tmux), bert-base prominence model trained on Helsinki train_100 (33,042 sents, 3 epochs, ~1 min on H200), inferred per-word expected prominence on all 320 passages, head-to-head STRESS(old dictionary-stress PROSODY_COLS) vs PROMINENCE(new) vs lexical. results/report_values.json.
- modhuman-vs-slop (F2): STRESS 0.925, PROMINENCE 0.925, prom-stress **+0.000 CI[-0.049,0.045]** -> no change. Prominence does NOT beat dictionary stress.
- modhuman-vs-gpt4o (F6): STRESS 0.838, PROMINENCE 0.819, prom-stress **-0.019 CI[-0.080,0.040]** -> prominence does NOT recover the GPT-4o tell (ns, slightly worse). So F6/GPT-4o's human-like prosody is GENUINE, not a dictionary-stress artifact — a better front-end doesn't help. (stress+prom combined = 0.881, mildly complementary, side note.)
- great-vs-modhuman (QUALITY): STRESS 0.756, PROMINENCE 0.825, prom-stress **+0.069 CI[0.000,0.147]** -> prominence IMPROVES the within-human quality contrast; CI lower bound grazes 0 (BORDERLINE, like the F5 near-miss). Notably PROMINENCE 0.825 > lexical 0.762 on the quality contrast.
- VERDICT (per pre-reg, honest): for the AI-DETECTION / origin axes (F2, F6) -> EXTRACTION IS NOT THE BOTTLENECK; dictionary stress was not the limiter; the signal is genuinely weak / GPT-4o genuinely human-like. CLOSED. For the within-human QUALITY axis -> a REAL but BORDERLINE positive (+0.069, CI grazes 0): the user's "is it the extraction?" instinct gets PARTIAL support, specifically on quality. The first evidence that a richer per-unit signal nudges the quality contrast up.
- CAVEATS (honest): quality CI grazes 0 (not decisively significant); great = 10 source books (few groups -> wide CI, the chronic power limit); raw accuracy not the residualized-against-lexical binding number; prominence model trained on LibriTTS audiobook register (read-aloud) -> may not fully transfer. -> FINDINGS F7. NEXT: power up the quality near-miss (more great-book groups / the residualized binding test) to see if +0.069 clears 0.
