#!/usr/bin/env python3
"""Phase-1c typography normalization (M13 fix).

Independent verification found a content-free typography classifier separates the
prose classes 0.83-0.96 (great-vs-modhuman 0.91) because the source registers carry
distinct glyph fingerprints that source-grouped CV and the lexical-slop residualizer
both miss:

  * great / modgreat (Gutenberg): curly quotes (“ ” ‘ ’), curly apostrophe-in-word,
    em-dashes.
  * modhuman (Fan-2018 / sacremoses detokenized): straight apostrophes, doubled
    straight-single dialogue marks ('' as a smart-quote stand-in), split contractions
    ("wo n't", "ca n't", generic "<verb> n't"), occasional space-before-punctuation,
    ":)" emoticons, "----" rule fragments.
  * slop (AI): a MIX of curly and straight (model-dependent).
  * flat (Reuters wire): no quotation marks at all.

`normalize_typography` canonicalizes ALL of these to one ASCII surface so no
punctuation/glyph feature can tell the classes apart, WITHOUT touching the words,
sentence boundaries, or stress-bearing tokens that the prosody features read. It is
applied UNIFORMLY to every pile (re-clean in place, same passage_ids), so the
carry-forward quarantine and the held-out are preserved.

Design rules that protect the prosody signal:
  * It only rewrites PUNCTUATION GLYPHS and detok WHITESPACE artifacts, never letters.
  * Split contractions are RE-JOINED ("wo n't" -> "won't"), which is what every other
    pile already has, so CMUdict/pronouncing tokenization and the spw/oov features are
    if anything MORE consistent across classes after normalization, not distorted.
  * Quotes are stripped to nothing (not converted to a straight glyph) so that the
    quote-character COUNT is identically zero in every pile. spaCy still segments
    sentences on the surviving terminal punctuation; quote glyphs were never alpha
    tokens, so they were never in `words`, `sl`, `spw`, or stress at all.
  * Em-dash / en-dash -> a spaced " - " (kept as a clause boundary token so ph_*
    clause-spacing stays comparable; the boundary set in features_lib treats - / – / —
    identically anyway, and now every pile uses the same one).
"""
import re
import unicodedata

# ---- contraction re-join (sacremoses split-token detok artifact) ------------------
# "wo n't" -> "won't", "ca n't" -> "can't", "did n't" -> "didn't", "ai n't" -> "ain't".
# Generic "<word> n't" join (covers is/was/were/has/had/should/could/would/do/does/...).
_NT_SPLIT = re.compile(r"\b([A-Za-z]+) n't\b")
# split clitics that sacremoses occasionally emits with a leading space:
# "they 're" -> "they're", "I 'm" -> "I'm", "we 'll" -> "we'll", "I 've" -> "I've",
# "he 'd" -> "he'd", "John 's" -> "John's". (Rare in this corpus but harmless to fix.)
_CLITIC_SPLIT = re.compile(r"\b([A-Za-z]+) '(re|m|ll|ve|d|s)\b")
# apostrophe split apart from both sides: "wo ' n't"/"can ' t"/"John ' s"
_APOS_TRIPLE = re.compile(r"\b([A-Za-z]+) ' ([A-Za-z]+)\b")

# ---- quote / glyph canonicalization -----------------------------------------------
# doubled straight singles used as a smart-quote stand-in -> remove (count -> 0).
_DOUBLE_SINGLE = re.compile(r"''+")
_DOUBLE_BACKTICK = re.compile(r"``+")
# any remaining quotation glyph (curly or straight, single or double) -> remove.
# NOTE: word-internal apostrophes are handled SEPARATELY first (see normalize_typography)
# so contractions/possessives keep a single canonical straight apostrophe.
_QUOTES = re.compile(r"[“”„‟‘’‚‛″′\"`]")

# reddit-style horizontal rule "- - - -" / "----" / "***" -> drop entirely (not prose).
_HRULE = re.compile(r"(?:\s[-*_]){3,}\s?|[-*_]{3,}")
# em/en-dash + ascii double-hyphen -> spaced single hyphen (one clause-boundary token).
_DASHES = re.compile(r"\s*(?:—|–|--+)\s*")

# ellipsis (unicode or spaced/anchored dots) -> single space (it is not stress-bearing).
_ELLIPSIS = re.compile(r"…|\.\s*\.\s*\.+")

# leftover markdown emphasis / code / heading / blockquote / rule artifacts.
_MD_EMPH = re.compile(r"[*_~`#>]+")
# emoticons that sacremoses leaves intact (":)", ":-(", ";)", etc.) -> remove.
_EMOTICON = re.compile(r"[:;=]['`]?[-^]?[)(DPp/\\|]+")

# whitespace fixes.
_NBSP = re.compile(r"[   ]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def normalize_typography(text: str) -> str:
    """Canonicalize punctuation/glyphs/detok artifacts to one ASCII surface.

    Letters, sentence-terminal punctuation, and token order are preserved, so the
    prosody features (stress, syllables, sentence-length, clause-spacing) are
    unchanged up to the re-joining of split contractions (which makes tokenization
    MORE consistent across piles, not less)."""
    if not text:
        return text

    # 0) Unicode NFKC: folds odd compatibility forms; turns &nbsp; etc. consistent.
    t = unicodedata.normalize("NFKC", text)
    t = _NBSP.sub(" ", t)

    # 1) re-join split contractions / clitics BEFORE quote stripping, so the
    #    apostrophe they need is preserved and we do not orphan an "n't".
    t = _APOS_TRIPLE.sub(r"\1'\2", t)
    t = _NT_SPLIT.sub(r"\1n't", t)
    t = _CLITIC_SPLIT.sub(r"\1'\2", t)

    # 2) protect word-internal apostrophes (curly or straight) -> canonical straight.
    #    Do this BEFORE the blanket quote strip so "don't"/"O'Brien"/"Anna's" survive
    #    with exactly one straight apostrophe, identical across every pile.
    t = re.sub(r"([A-Za-z])[’ʼ'‘]([A-Za-z])", r"\1'\2", t)

    # 3) collapse the doubled-single / doubled-backtick dialogue marks.
    t = _DOUBLE_SINGLE.sub(" ", t)
    t = _DOUBLE_BACKTICK.sub(" ", t)

    # 4) dashes, ellipsis, emoticons, markdown.
    t = _HRULE.sub(" ", t)
    t = _DASHES.sub(" - ", t)
    t = _ELLIPSIS.sub(" ", t)
    t = _EMOTICON.sub(" ", t)
    t = _MD_EMPH.sub(" ", t)

    # 5) strip ALL remaining quotation glyphs (now only sentence-delimiting quotes
    #    remain; word-internal apostrophes were already canonicalized in step 2).
    t = _QUOTES.sub("", t)

    # 6) whitespace cleanup: drop space-before-punctuation, collapse runs.
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _MULTISPACE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


if __name__ == "__main__":
    # quick self-check on the documented artifacts.
    samples = [
        ("modhuman split-nt", "she ca n't hear it and I did n't move"),
        ("modhuman dialogue", "called High Notes,'' because he focused"),
        ("modhuman clitic", "they 're safe but you wo n't know"),
        ("great curly", "“It would have been,” he said—softly—‘really’"),
        ("great apos", "could not’ve said why; Madame Olenska’s talk"),
        ("emoticon/rule", "liked the promt. :) - - - - - - - So yeah"),
    ]
    for name, s in samples:
        print(f"{name:22s} {s!r}\n{'':22s} -> {normalize_typography(s)!r}\n")
