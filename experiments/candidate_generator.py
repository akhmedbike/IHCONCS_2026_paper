"""Candidate generation for correction, in three ranking modes.

The paper (Table 3 / eq. 7) compares candidate-generation strategies. This
module exposes a single `generate_candidates(token, mode)` that returns a
ranked list of (candidate, score) pairs, where `score` is comparable within a
mode (lower energy = better). Three modes are implemented, all sharing the
same SymSpell retrieval (kazmorph.suggestor._get_symindex):

  * "dl"           — pure Damerau-Levenshtein retrieval, ranked by
                     (distance, alphabetical). No frequency, no morphology.
                     This is the [1], [2] baseline.

  * "dl_freq"      — same retrieval, re-ranked by (distance, -frequency,
                     alphabetical). The corpus-frequency tie-break is the
                     data-driven baseline [3].

  * "dl_fst_filter" — retrieval, then drop any candidate whose reverse
                     morphological analysis fails or whose compatibility
                     coefficient κ < KAPPA_MIN. This is the analysis-side
                     approximation of the paper's analysis–synthesis–reanalysis
                     verification (eq. 7); the full synthesiser is specified
                     but not implemented, and this is disclosed in the paper.

  * "dl_freq_morph" — dl_freq plus a stem-recombination channel: the token is
                     morphologically decomposed (RMA), near-candidates for the
                     *stem* are retrieved, and the suffix is re-attached (with
                     harmony alternates) to synthesise surface candidates the
                     edit-distance index cannot reach. This is the implemented
                     first step toward the specified morphological synthesiser;
                     it is analysis-side recombination, not full synthesis.
                     Empirical lift: candidate reachability +11 pp on fold 1.

Every mode returns at most MAX_CANDIDATES pairs. The candidate list never
includes the input token itself — the caller adds the "keep original" option
separately so that token preservation is a first-class decision (paper §"Error
Detection and Candidate Ranking", last paragraph before eq. 14).

Leakage note: a naive "grow the index with corpus surface forms" path was
evaluated and rejected — the frequency dictionary is built from the KTB train
split, so its inflected forms ARE the gold test-fold forms, and including
them makes candidate recall trivially 95% but circular. The leakage-free
augmented index (test-fold forms excluded) gives 0% lift, because Kazakh
agglutinative morphology produces near-unique surface forms per suffix
combination. Stem recombination is the only leakage-free way to recover
inflected forms whose stem is in the lexicon.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from experiments.freq import get_frequency_dict, frequency_of  # noqa: E402

# Reuse the kazmorph SymSpell index rather than rebuilding one. The index is a
# process-wide singleton, so importing it once is cheap and keeps memory flat.
from kazmorph.suggestor import _get_symindex  # noqa: E402
from kazmorph.rma import ReverseMorphAnalyzer, RMAResult  # noqa: E402
from kazmorph.mfrn import MFRN  # noqa: E402


# Module-level singletons for the FST-filter mode. ReverseMorphAnalyzer loads
# the UD lemma dict on construction; MFRN is stateless beyond its rule table.
# Both are safe to share across folds and threads (read-only after __init__).
_RMA: ReverseMorphAnalyzer | None = None
_MFRN: MFRN | None = None
_FREQ: dict[str, int] | None = None


def _get_rma() -> ReverseMorphAnalyzer:
    global _RMA
    if _RMA is None:
        _RMA = ReverseMorphAnalyzer()
    return _RMA


def _get_mfrn() -> MFRN:
    global _MFRN
    if _MFRN is None:
        _MFRN = MFRN(scoring=False)
    return _MFRN


def _get_freq() -> dict[str, int]:
    global _FREQ
    if _FREQ is None:
        _FREQ = get_frequency_dict()
    return _FREQ


def _symspell_retrieve(token: str, idx=None) -> list[tuple[int, str]]:
    """Raw SymSpell retrieval: [(distance, word)] within MAX_EDIT_DISTANCE.

    Lowercases the token to match the index convention. The result is NOT
    sorted; each mode applies its own sort key. `idx` defaults to the
    process-wide singleton (_get_symindex); a caller may pass a fold-specific
    index built with the test-fold forms excluded (leakage guard).
    """
    if idx is None:
        idx = _get_symindex()
    return idx.query(token.lower())


def _fst_valid(candidate: str) -> bool:
    """Analysis-side morphological validity check (the "re-analysis" proxy).

    A candidate is valid iff ReverseMorphAnalyzer recovers a non-UNK analysis
    for it AND the MFRN compatibility coefficient κ ≥ KAPPA_MIN. This is an
    approximation of eq. (7)'s `Analyze(c) ⊇ (ℓ′, f′)`: we cannot check the
    full lemma+feature reproduction without a synthesiser, so we check that
    the candidate at least parses into a known stem with admissible phonology.
    """
    rma = _get_rma()
    mfrn = _get_mfrn()
    res = rma.analyze(candidate)
    # UNK means no stem and no suffixes recovered → structurally inadmissible.
    if res.pos == "UNK" and not res.suffixes:
        return False
    kappa = mfrn.compute_kappa(res)
    return kappa >= config.KAPPA_MIN


# ── Stem-expansion channel (dl_freq_morph mode) ──────────────────────────────
def _suffix_variants(suffixes: list[str]) -> list[list[str]]:
    """Return harmony-alternate suffix sequences for re-synthesis.

    Kazakh vowel harmony selects suffix allomorphs by stem backness. A
    corrupted token may carry the wrong allomorph (the harmony-violation
    operator) or an analysis of the corrupted form may have picked the
    back/front variant that does not match the gold. To recover the gold via
    stem recombination we therefore generate, alongside the suffixes observed
    on the corrupted token, their harmony counterparts.

    The mapping covers the productive vowel-alternating suffix initials:
    back ↔ front pairs (а/е, ы/і, о/ө, ұ/ү). This is intentionally a small,
    linguistically-grounded table rather than a full FST — it targets exactly
    the allomorph mismatches that the controlled-error operators introduce.
    """
    _HARMONY = {"а": "е", "ы": "і", "о": "ө", "ұ": "ү",
                "е": "а", "і": "ы", "ө": "о", "ү": "ұ"}
    variants = [list(suffixes)]
    for i, suf in enumerate(suffixes):
        if suf and suf[0] in _HARMONY:
            alt = [_HARMONY[suf[0]] + suf[1:]]
            variants.append(list(suffixes[:i]) + alt + list(suffixes[i + 1:]))
    return variants


def _stem_expand(token: str, idx) -> list[tuple[str, int]]:
    """Stem-recombination retrieval: return [(candidate, edit_distance)].

    Decomposes `token` into stem + suffixes via RMA, retrieves near-candidates
    for the stem in `idx`, and re-synthesises surface candidates by
    re-attaching the suffixes (and their harmony alternates). The returned
    edit distance is that of the *stem* match, since the suffix is carried
    over verbatim — this is the key difference from direct retrieval, which
    would require the full inflected form to be in the index.

    Returns [] when the token cannot be decomposed (RMA returns UNK or no
    suffixes) or when no stem candidate is found. The caller applies
    STEM_PENALTY to the energy of these candidates so they rank below
    same-distance direct hits.
    """
    rma = _get_rma()
    res = rma.analyze(token)
    # Need a real decomposition: non-UNK, with suffixes, and a stem shorter
    # than the full token (otherwise there is nothing to recombine).
    if res.pos == "UNK" or not res.suffixes or res.lemma == token:
        return []
    stem = res.lemma
    stem_hits = idx.query(stem.lower())
    if not stem_hits:
        return []
    # Re-synthesise: for each near-stem, attach each suffix variant.
    variants = _suffix_variants(res.suffixes)
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    t_lower = token.lower()
    for dist, stem_cand in stem_hits[:config.STEM_MAX_CANDIDATES]:
        for suf_seq in variants:
            surface = stem_cand + "".join(suf_seq)
            if surface == t_lower or surface in seen:
                continue
            seen.add(surface)
            out.append((surface, dist))
    return out


def generate_candidates(
    token: str,
    mode: str = "dl",
    max_candidates: int | None = None,
    *,
    index=None,
    freq: dict[str, int] | None = None,
) -> list[tuple[str, float]]:
    """Return ranked [(candidate, energy)] for `token` under the given mode.

    Energy is the ranking key from eq. (13), with λ_ctx = 0 (no trained
    contextual ranker). Lower energy = better candidate. The list is sorted
    ascending by energy and truncated to `max_candidates` (default
    config.MAX_CANDIDATES). The input token itself is excluded.

    `mode` ∈ {"dl", "dl_freq", "dl_fst_filter", "dl_freq_morph"}.

    The "dl_freq_morph" mode is dl_freq plus a stem-recombination channel:
    the corrupted token is morphologically decomposed (RMA), near-candidates
    for the *stem* are retrieved, and the suffix is re-attached (with harmony
    alternates) to synthesise surface candidates the edit-distance index
    cannot reach. This is the analysis-side approximation of the paper's
    morphological synthesiser (specified, not fully implemented).

    `index` optionally supplies a SymSpell index (e.g. a fold-specific one
    with test-fold forms excluded); it defaults to the global singleton.

    `freq` optionally supplies a frequency dictionary (e.g. a leakage-free
    per-fold dict built by experiments.freq.build_fold_frequency_dict); it
    defaults to the global singleton. Passing a per-fold dict removes the
    gold-form leakage that the global dict carries into the dl_freq /
    dl_freq_morph ranking (see experiments.freq docstring).
    """
    if max_candidates is None:
        max_candidates = config.MAX_CANDIDATES

    idx = index if index is not None else _get_symindex()
    raw = _symspell_retrieve(token, idx)
    freq = freq if freq is not None else _get_freq()

    # First pass: build (candidate, distance, freq) for surviving candidates.
    survivors: list[tuple[str, int, int, bool]] = []  # (word, dist, freq, is_stem)
    seen: set[str] = set()
    t_lower = token.lower()
    for dist, word in raw:
        if word == t_lower or word in seen:
            continue
        # dl_fst_filter / dl_freq_morph apply the morphological validity gate
        # here, before any scoring. Candidates that fail the gate are removed
        # entirely (paper: "Morphologically invalid candidates are removed
        # before neural ranking rather than receiving only a soft penalty").
        if mode in ("dl_fst_filter", "dl_freq_morph") and not _fst_valid(word):
            continue
        seen.add(word)
        survivors.append((word, dist, frequency_of(word, freq), False))

    # Stem-expansion channel: only for dl_freq_morph. Produces additional
    # surface candidates by recombining near-stems with the token's suffixes.
    # These are tagged is_stem=True so the scorer applies STEM_PENALTY.
    if mode == "dl_freq_morph":
        for word, dist in _stem_expand(token, idx):
            if word == t_lower or word in seen:
                continue
            # Apply the same morphological validity gate to recombined forms.
            if not _fst_valid(word):
                continue
            seen.add(word)
            survivors.append((word, dist, frequency_of(word, freq), True))

    if not survivors:
        return []

    # Score every survivor with the mode-appropriate energy.
    scored: list[tuple[str, float]] = []
    for word, dist, f, is_stem in survivors:
        # Eq. (13) with λ_ctx = 0. The morphological-uncertainty term
        # Ω_morph(c) is (1 - κ); for dl / dl_freq modes (no MFRN pass) it is
        # constant across candidates and drops out of the argmin, so we only
        # compute it for dl_fst_filter / dl_freq_morph where κ varies.
        if mode in ("dl_fst_filter", "dl_freq_morph"):
            kappa = _get_mfrn().compute_kappa(_get_rma().analyze(word))
            omega_morph = 1.0 - kappa
        else:
            omega_morph = 0.0
        normalised_dist = dist / config.MAX_EDIT_DISTANCE
        energy = (
            config.LAMBDA_ED * normalised_dist
            + config.LAMBDA_MORPH * omega_morph
            - config.LAMBDA_FREQ * (1.0 if f > 0 else 0.0)  # binary presence bonus
        )
        # Frequency as a fine-grained tie-break: higher freq → lower energy.
        # We add a small log term so two same-distance candidates are ordered
        # by corpus frequency, realising the dl_freq ranking.
        if mode in ("dl_freq", "dl_freq_morph"):
            import math
            energy -= 0.1 * math.log(1 + f)
        # Stem-recombined candidates carry an extra penalty: their distance is
        # measured on the stem only, so they are less reliable than a direct
        # full-form edit-distance hit at the same band.
        if is_stem:
            energy += config.STEM_PENALTY
        scored.append((word, energy))

    scored.sort(key=lambda x: (x[1], x[0]))
    return scored[:max_candidates]


def candidate_probs(
    candidates: list[tuple[str, float]],
    include_original: bool = True,
    original_token: str = "",
) -> list[tuple[str, float]]:
    """Normalised softmax probabilities over candidates (eq. 14).

    P(c | X, i) = exp(-E(c)) / Σ exp(-E(c′)), with the original token added
    to the decision set as its own zero-energy option when include_original
    is set. Returns [(candidate, probability)] sorted by descending probability.

    The original-token-as-candidate is what makes KEEP a first-class model
    decision rather than a fallback (paper, end of eq. 14 paragraph).
    """
    import math
    items = list(candidates)
    if include_original and original_token:
        # Give the original token a neutral energy (0.0). It competes on equal
        # footing with the corrections; if no correction has lower energy,
        # KEEP wins on the softmax. This is intentionally not a bonus — we do
        # not want to bias toward inaction beyond what the evidence supports.
        items = [(original_token, 0.0)] + items
    if not items:
        return []
    # Softmax over -E. Subtract max for numerical stability.
    neg_e = [-e for _, e in items]
    m = max(neg_e)
    exps = [math.exp(ne - m) for ne in neg_e]
    total = sum(exps)
    probs = [e / total for e in exps]
    paired = [(word, p) for (word, _), p in zip(items, probs)]
    paired.sort(key=lambda x: -x[1])
    return paired
