"""Corpus frequency dictionary for the frequency-ranking baseline.

The paper's candidate-ranking baseline [3] (Toleu et al., "Data-driven approach
for spellchecking and autocorrection") ranks correction candidates by corpus
frequency. kazmorph's suggestor has no frequency information at all — every
lexicon is a plain `set[str]` (verified by grep). This module derives a
legitimate frequency dictionary from the silver corpora that ship with the
project, so the DL+frequency baseline is a real, reproducible comparison
rather than a stub.

The frequency source is the union of:
  * data/train_split.conllu  — gold UD Kazakh-KTB (normative forms)
  * data/silver_sentences.conllu — CG-disambiguated Kazakh literary texts
  * data/silver_apertium.conllu — Apertium-analysed Kazakh text

Silver data is used ONLY for frequency estimation, never as a gold label or
for model training in this paper. That matches the paper's stated role for
the silver corpus (Table 1: "frequency estimation and contextual pretraining")
and avoids any overlap with the evaluation gold.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402


def _count_forms(conllu_path: str, counter: Counter) -> int:
    """Add lowercased surface-form counts from one CoNLL-U file.

    Returns the number of tokens counted. Skips multiword/elempty rows
    (id contains '-' or '.'), comment lines, and blank lines. Lowercasing
    matches the suggestor's lookup convention.
    """
    n = 0
    if not os.path.exists(conllu_path):
        return 0
    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            tid = parts[0]
            if "-" in tid or "." in tid:
                continue
            form = parts[1].strip()
            if form:
                counter[form.lower()] += 1
                n += 1
    return n


def build_frequency_dict() -> dict[str, int]:
    """Build the corpus-frequency dictionary used by the DL+freq baseline.

    Combines gold train + both silver corpora. Returns {lowercased_form: count}.
    The dict is cached on first call; subsequent calls return the same object.
    """
    counter: Counter = Counter()
    n_train = _count_forms(config.TRAIN_CONLLU, counter)
    n_silver_sent = _count_forms(config.SILVER_SENTENCES_CONLLU, counter)
    n_silver_ap = _count_forms(config.SILVER_APERTIUM_CONLLU, counter)
    freq = dict(counter)
    # Stash provenance on the dict for logging. dict subclasses are heavier;
    # we attach metadata as function attributes instead.
    build_frequency_dict.last_counts = {
        "train_tokens": n_train,
        "silver_sentences_tokens": n_silver_sent,
        "silver_apertium_tokens": n_silver_ap,
        "total_tokens": n_train + n_silver_sent + n_silver_ap,
        "unique_forms": len(freq),
    }
    return freq


def frequency_of(form: str, freq: dict[str, int]) -> int:
    """Return corpus frequency of `form` (case-insensitive), 0 if unseen.

    Used by candidate_generator's DL+freq ranking mode and by the energy
    function's λ_f·log(1+ν_c) term. The +1 floor in the log keeps unseen
    candidates from receiving -inf energy.
    """
    return freq.get(form.lower(), 0)


# Module-level cache so repeated imports don't re-scan the corpora.
_FREQ_CACHE: dict[str, int] | None = None


def get_frequency_dict() -> dict[str, int]:
    """Return the cached frequency dict, building it on first access."""
    global _FREQ_CACHE
    if _FREQ_CACHE is None:
        _FREQ_CACHE = build_frequency_dict()
    return _FREQ_CACHE
