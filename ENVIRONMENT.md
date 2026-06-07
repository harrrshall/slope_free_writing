# Environment

> Setup, libraries, data sources, and exact commands for the prosody experiments.
> Phase 0 (the separation pre-test) is the only phase wired up here. Everything was
> probed on this machine; figures below are measured, not assumed. Append, never silently
> rewrite, when the stack changes.

---

## 0. Machine facts (probed this session)

- Python 3.12.3 (`python3`; there is no bare `python` on PATH; always call `python3`).
- PEP 668 marker present (`/usr/lib/python3.12/EXTERNALLY-MANAGED`), so plain `pip` is
  refused. Install with `--break-system-packages` (per AGENTS.md §6). Installs land in
  `~/.local`. Do **not** `sudo pip`; `sudo` needs a password here.
- `eSpeak` 1.48.15 and `eSpeak NG` 1.51 are already installed (`/usr/bin/espeak`,
  `/usr/bin/espeak-ng`); `/lib/x86_64-linux-gnu/libespeak-ng.so.1` present. `prosodic`/
  `phonemizer` find it automatically. No `apt` step needed. Used ONLY for text->phoneme/
  stress G2P on out-of-dictionary words, **never** audio synthesis (honors D4 / M6).
- `nltk_data` already contains `corpora/reuters.zip` (10,788 docs, verified loadable) and
  `tokenizers/punkt`, `tokenizers/punkt_tab`. Still missing: `cmudict`,
  `averaged_perceptron_tagger_eng`.
- Already in `~/.local`: numpy 2.4.2, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.17.1,
  pronouncing 0.3.0, cmudict 1.1.3, nltk 3.9.4, prosodic 3.3.0.
- Missing, must install: `spacy` (+ `en_core_web_sm` model), `datasets`, `sacremoses`.
- Network reachable: pypi.org, files.pythonhosted.org, huggingface.co, gutenberg.org,
  github.com (release assets).
- `texts/1342.txt.gz` (Gutenberg #1342, Pride & Prejudice) is already on disk.

### Repo layout note (logged, do not silently relocate)
AGENTS.md references `docs/ENVIRONMENT.md` and `docs/*`, but there is **no `docs/`
subdirectory**; all docs live at the repo root. This file is the root `ENVIRONMENT.md`,
the one that exists. Writing a second copy under `docs/` would split the source of truth.
If the `docs/` layout is ever adopted, do it as one move and log it in DECISIONS.md.

---

## 1. Install (virtualenv is the REQUIRED path)

The user requires a project virtualenv, NOT `pip --break-system-packages`. Create `.venv`
at the repo root and install the full stack into it. Run every Phase-0 script with
`.venv/bin/python3`.

```bash
cd /home/cybernovas/Desktop/2026/experiments/slope_free_writing
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
  prosodic pronouncing cmudict nltk spacy datasets sacremoses gutenbergpy \
  scikit-learn numpy pandas scipy
```

Inside a venv do NOT pass `--break-system-packages`. `gutenbergpy` (Gutenberg fetch for the
GREAT pile) is required and easy to forget; it is in the list above. All later scripts assume
`.venv/bin/python3`.

---

## 2. Post-install data resources

```bash
# NLTK: cmudict + POS tagger. reuters is already present (do not re-fetch).
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ('cmudict','averaged_perceptron_tagger_eng')]"

# spaCy English model (sentence + clause segmentation, per D5). Inside the venv this works
# directly (no PEP-668 block); the bare 'python3 -m spacy download' OUTSIDE a venv fails on
# this box because spacy shells out to pip without --break-system-packages.
.venv/bin/python -m spacy download en_core_web_sm
```

On nltk >= 3.9 the tagger resource is `averaged_perceptron_tagger_eng` (the legacy
`averaged_perceptron_tagger` name raises a LookupError pointing to the `_eng` variant).
If GitHub rate-limits the spaCy model, fall back to the direct wheel:
```bash
.venv/bin/pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

---

## 3. Pinned versions & roles

| Library | Version (this box) | Role in Phase 0 |
|---|---|---|
| prosodic | 3.3.0 | Metrical-tension proxy (constraint-based scansion). v3 API: `Text(s).parse()` -> `line.best_parse` -> `.score` / `.num_sylls` / `.num_viols` / `.meter_str` / `.stress_str` (properties, NOT methods). OOV handled internally via eSpeak (`lang_espeak='en-us'`). SLOWEST feature; added only as the rescue round (see protocol). |
| pronouncing | 0.3.0 | CMUdict stress + syllable counts. `phones_for_word` returns a LIST ([] = OOV); `stresses(s)` == `re.sub(r'[^012]','',s)` (0=unstressed,1=primary,2=secondary); `syllable_count(phones)` == `len(stresses(phones))`. |
| cmudict | 1.1.3 | Backing dictionary (pronouncing initializes it). ~134k North-American entries; British/archaic spellings raise OOV. |
| spacy + en_core_web_sm | 3.8.14 / 3.8.0 | Sentence segmentation (`doc.sents`), clause/phrase boundaries via dep + punctuation. |
| nltk | 3.9.4 | `reuters` corpus (FLAT pile); cmudict/tagger data. |
| datasets | (install) | HF loader for the AI-slop and modern-human piles. |
| sacremoses | (install) | `MosesDetokenizer` for the space-tokenized euclaise text (M5 fix). |
| scikit-learn | 1.9.0 | LogisticRegression, DecisionTree, StratifiedKFold / StratifiedGroupKFold, permutation_test_score, permutation_importance. |
| scipy | 1.17.1 | `stats.t.interval` (fold-wise CI), `stats.skew`. |
| numpy / pandas | 2.4.2 / 3.0.3 | feature matrix + manifest. |

Pin `prosodic==3.3.0` and `pronouncing==0.3.0`: the v3 prosodic API differs from older v1
docs (`best_parse`/`score`, properties not methods). If a future install drifts, the
feature extractor breaks.

---

## 4. Data sources (verified this session)

| Pile | Source | Fetch | Verified | Caveats |
|---|---|---|---|---|
| GREAT (classic anchor) | Project Gutenberg via `gutenbergpy` 0.3.5 | `import gutenbergpy.textget as tg; tg.strip_headers(tg.get_text_by_id(ID))` | yes | `strip_headers` removes ONLY the PG license wrapper, NOT book-internal intro/TOC/[Illustration]/footnotes; needs head/tail trim + heading filter (see protocol). Pre-1928 => era confound. |
| FLAT (human) | NLTK `reuters` (Reuters-21578) | already on disk; `from nltk.corpus import reuters` | yes | 10,788 docs; **19.7% fall in [150,400] words (788 of first 4000; 2168 overall)**; ample. Strip the leading ALL-CAPS headline line. Finance/commodity topic skew -> sample across categories. 1987 newswire (its own era). |
| AI-SLOP | HF `lars1234/story_writing_benchmark`, config `average`, split `train` | `load_dataset('lars1234/story_writing_benchmark','average',split='train')` | yes | MIT. Fields `story_text`, `overall_score`, `language`, `model_name`. **Config is `average` (NOT `default`)**. **Multilingual en/de/es -> MUST filter `language=='en'`**. The `average` file repeats stories (~8,520 unique can expand to ~42,600 rows) -> de-dup on `story_text` and log actual N before computing the score quartile. `overall_score` is an LLM-judge metric (slop-biased, M2) -> OK to DEFINE the slop tail, never reuse as a held-out judge. |
| MODERN-HUMAN (Phase-1 `modhuman` pile) | HF `euclaise/writingprompts`, split `train` | `load_dataset('euclaise/writingprompts', split='train')` | yes | Fields `prompt`, `story`. **PROVENANCE RESOLVED (Phase 1, D10):** this is the FROZEN Fan et al. 2018 FAIR WritingPrompts release (arXiv:1805.04833, MIT, 303,358 rows), scraped from r/WritingPrompts years before GPT-2 (2019) and ChatGPT (2022), so it is **human BY CONSTRUCTION** and **M4-admissible** as competent-modern-human prose. The earlier "not verifiable-human" worry applies only to a FRESH live re-scrape, NOT this frozen archive; do not re-scrape live. It is competent-amateur, NOT acknowledged-great. Register-clean (strip markdown/NSFW-tags/`&nbsp;`/URLs) and detokenize (`wo n't`, ` . `) before feature extraction so forum typography is not a new register confound. |

### Rejected / backup
- `agentlans/literary-genre-examples`; **REJECTED for any human pile**: card states "Created
  by Claude Sonnet 4". AI text in a human pile violates M4.
- `lars76/story-evaluation-llm`; does not exist (HTTP 401); the correct id is
  `lars1234/story_writing_benchmark`.
- `sam-paech/slop-forensics` (GitHub, MIT); AI-slop BACKUP only; ships NO pre-generated
  stories (needs an API key to generate). Its `slop_lists` are the Phase-1 lexical baseline.
- `EQ-bench/creative-writing-bench`; ships prompts + zipped run artifacts, not a tidy
  labeled corpus. Provenance/cross-check only; `lars1234` is the primary AI-slop source.

---

## 5. Smoke test (run after install + data download)

```bash
python3 - <<'PY'
import pronouncing, nltk, spacy, sklearn, prosodic, itertools, statistics
# 1) CMUdict stress
print('stress(rhythm):', pronouncing.stresses(pronouncing.phones_for_word('rhythm')[0]))
# 2) spaCy segmentation
nlp = spacy.load('en_core_web_sm')
print('sents:', [s.text for s in nlp('The prose breathes. It does not.').sents])
# 3) prosodic v3 scansion (exercises eSpeak OOV path; properties, not methods)
t = prosodic.Text('The silent reader hears a rhythm in the prose tonight.'); t.parse()
bp = t.lines[0].best_parse
print('prosodic score/num_sylls:', bp.score, bp.num_sylls, '| meter:', bp.meter_str)
# 4) M3 guard sanity: true metronome '01'*N must be ~0 on every variation feature
seq='01'*120
idx=[i for i,c in enumerate(seq) if c=='1']; gaps=[b-a for a,b in zip(idx,idx[1:])]
runs=[len(list(g)) for _,g in itertools.groupby(seq)]
gcv=statistics.pstdev(gaps)/statistics.mean(gaps); rv=statistics.pvariance(runs)
print('metronome gap_cv/run_len_var (must be ~0):', round(gcv,4), round(rv,4))
assert gcv < 1e-9 and rv < 1e-9, 'M3 metronome baseline broken'
print('SMOKE OK')
PY
```

Expected: a stress string (`01`), a 2-sentence split, a non-empty prosodic parse, the
metronome line printing `0.0 0.0`, and `SMOKE OK`. The first `prosodic.Text` call is slow
(warms the phonemizer/eSpeak backend); expected, not a failure.

---

## 6. Permission blockers

None for Phase 0. `sudo`/`apt` cannot run unattended (password required), but every system
dependency (eSpeak, libespeak-ng) is already present, and every Python package installs to
`~/.local` via `--break-system-packages`. The only network dependencies are the HF datasets
and the Gutenberg/spaCy fetches, all reachable this session.