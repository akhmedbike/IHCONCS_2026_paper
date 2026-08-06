"""
Stage 3b: Neural lemmatizer.

A KazBERT token classifier that predicts, for each word, the length of the
lemmatic prefix (how many leading characters to keep). This replaces the
rule-based RMA fallback for words not in the lemma dictionary, where RMA
under-strips ~82% of the time.

Architecture mirrors kazmorph/neural.py (POS tagger) so the load/predict/stub
scaffolding is identical; only the head's label semantics differ (prefix length
0..MAX_PREFIX instead of BIO POS tags) and a small post-processing step repairs
stem-final consonant alternations (қ↔ғ, к↔г, б↔п, д↔т, ж↔ш, з↔с) that the
length-only label cannot encode.

When the model file is absent, tag() passes tokens through unchanged, so the
pipeline degrades gracefully to the original RMA lemma (soft failure).
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from kazmorph.device import pick_device
from kazmorph.rma import RMAResult


# Match MAX_PREFIX in scripts/train_lemma.py so label decoding is consistent.
MAX_PREFIX = 24
NUM_LABELS = MAX_PREFIX + 1

# Kazakh final-consonant voicing/devoicing. The dictionary lemma usually keeps
# the voiceless base form (бастық, аймақ, орын…), while the surface form often
# carries the voiced alternant before a vowel-initial suffix (бастығ-ы,
# аймағ-ы). When the predicted cut leaves a final consonant that has a
# voiceless counterpart, we devoice it to recover the base lemma.
_DEVOICE = {"б": "п", "г": "к", "ғ": "қ", "д": "т", "ж": "ш", "з": "с"}
_VOICE = {"п": "б", "к": "г", "қ": "ғ", "т": "д", "ш": "ж", "с": "з"}
_VOWELS = set("аәеёиіоөуұүыэюя")


def _repair_lemma(form: str, keep_len: int) -> str:
    """
    Build lemma = form[:keep_len] with a final-consonant devoice repair.

    keep_len == 0 (or beyond word length) means "no change" → return the whole
    word, which is correct for uninflected forms (PROPN, PUNCT, particles).

    Devoicing fires ONLY when the character immediately after the cut is a
    VOWEL. The reason: surface forms carry the voiced alternant before a
    vowel-initial suffix (бастығ-ы, аймағ-ы), so a voiced consonant right
    before a vowel suffix signals the dictionary lemma wants the voiceless
    counterpart (бастық, аймақ). When the next char is a consonant or the cut
    is at the word end, the voiced consonant is the genuine lemma ending
    (сіз, қаз) and must be left alone — over-aggressive devoicing there turns
    correct predictions (сізде→сіз) into wrong ones (сізде→сіс).
    """
    if keep_len <= 0 or keep_len > len(form):
        return form
    lemma = form[:keep_len]
    # Fire only when there is a following char AND it is a vowel.
    if keep_len < len(form):
        next_ch = form[keep_len]
        if next_ch in _VOWELS and lemma and lemma[-1] in _DEVOICE:
            lemma = lemma[:-1] + _DEVOICE[lemma[-1]]
    return lemma


class NeuralLemmatizer:
    """Stage 3b: predict lemma via prefix-length token classification."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        if model_path and os.path.exists(os.path.join(model_path, "config.json")):
            self._load_model(model_path)

    def _load_model(self, path: str):
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Unified device selector (cuda > mps > cpu); see kazmorph.device and
        # neural.py for the rationale. ~2.7x speedup on Apple Silicon via MPS.
        self.device = pick_device()
        self.model.to(self.device)
        self.model.eval()

    def tag(self, tokens: list[RMAResult]) -> list[RMAResult]:
        """Predict and rewrite each token's lemma in place, or pass through."""
        if self.model is None:
            return tokens
        return self._predict(tokens)

    def _predict(self, tokens: list[RMAResult]) -> list[RMAResult]:
        """
        Predict lemmas for all tokens, chunking long inputs (same reason as the
        POS NeuralTagger: the model max_length=128 would otherwise silently drop
        every word past position ~100 in a long document).
        """
        if not tokens:
            return tokens

        CHUNK_WORDS = 100
        _SENT_END = {".", "!", "?", "»", "…"}
        boundaries = [i for i, t in enumerate(tokens)
                      if t.token in _SENT_END or any(c in t.token for c in _SENT_END)]
        boundaries.append(len(tokens) - 1)

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < len(tokens):
            ideal_end = start + CHUNK_WORDS
            if ideal_end >= len(tokens):
                chunks.append((start, len(tokens)))
                break
            best_end = ideal_end
            for b in boundaries:
                if start < b <= ideal_end:
                    best_end = b + 1
            if best_end <= start + 1:
                best_end = ideal_end
            chunks.append((start, best_end))
            start = best_end

        for c_start, c_end in chunks:
            self._predict_chunk(tokens, c_start, c_end)
        return tokens

    def _predict_chunk(self, tokens: list[RMAResult], start: int, end: int) -> None:
        """Predict lemmas for tokens[start:end] in one forward pass."""
        chunk = tokens[start:end]
        words = [t.token for t in chunk]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = logits.argmax(dim=-1)[0].tolist()

        # Align each word to its first subword prediction (same strategy as the
        # POS NeuralTagger).
        seen: set[int] = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            form = words[wid].lower()
            keep_len = int(predictions[idx])
            # Punctuation / numbers / short tokens: keep the RMA lemma.
            if not any(c.isalpha() for c in form):
                continue
            predicted_lemma = _repair_lemma(form, keep_len)
            # Only overwrite if we produced something non-empty and at least as
            # plausible as the RMA lemma (avoid clobbering good dictionary hits
            # on very short stems).
            if predicted_lemma and len(predicted_lemma) >= 2:
                chunk[wid].lemma = predicted_lemma
