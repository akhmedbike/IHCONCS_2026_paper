"""
Central configuration for the IHCONCS 2026 correction experiments.

All paths, hyperparameters, and seeds live here so every experiment script
imports the same values. This is the single source of truth that the paper's
"Implementation details" and "Notation and hyperparameter values" tables are
derived from — do not change a value here without updating the manuscript.
"""

from __future__ import annotations

import os

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolve relative to this file so the scripts run from any cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
ERRORS_DIR = os.path.join(RESULTS_DIR, "controlled_errors")

# Gold treebank (UD Kazakh-KTB, vendored copy).
TRAIN_CONLLU = os.path.join(DATA_DIR, "train_split.conllu")
DEV_CONLLU = os.path.join(DATA_DIR, "dev_split.conllu")
TEST_CONLLU = os.path.join(DATA_DIR, "test.conllu")
# Silver corpora (used only for the corpus-frequency baseline, never as gold).
SILVER_SENTENCES_CONLLU = os.path.join(DATA_DIR, "silver_sentences.conllu")
SILVER_APERTIUM_CONLLU = os.path.join(DATA_DIR, "silver_apertium.conllu")

# Trained models (restored kazmorph-2026 checkpoints, no retraining here).
NEURAL_POS_PATH = os.path.join(MODELS_DIR, "neural_pos")
NEURAL_LEMMA_PATH = os.path.join(MODELS_DIR, "neural_lemma")
CRF_MODEL_PATH = os.path.join(MODELS_DIR, "crf_model.joblib")


def ensure_dirs() -> None:
    """Create all output directories. Idempotent; safe to call at import."""
    for d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR, RAW_DIR, ERRORS_DIR):
        os.makedirs(d, exist_ok=True)


# ── Cross-validation ─────────────────────────────────────────────────────────
# The paper originally claimed 10-fold CV, but the shipped harness implements
# 5-fold stratified CV. We report 5-fold honestly and state the reduction in
# Section III. 1078 sentences is too small for the difference to matter
# statistically, but it must be disclosed.
N_FOLDS = 5
CV_SEED = 42

# Number of corruption-seed repetitions per fold. The paper claimed 5 random
# initializations; we run 3 (the deterministic parts — coverage, FCR on clean
# text — are seed-independent and effectively single-run).
N_CORRUPTION_SEEDS = 3
CORRUPTION_SEEDS = [13, 42, 123]

# Fraction of identity (unchanged) sentences in the controlled error set.
# Must be >= 0.30 to keep the false-correction rate measurable; if too few
# clean sentences are present, FCR has no denominator.
IDENTITY_FRACTION = 0.35
# Per-sentence corruption probability (1 - IDENTITY_FRACTION applies at the
# sentence level: with prob 0.65 we corrupt exactly one token in the sentence).
CORRUPT_PROB = 1.0 - IDENTITY_FRACTION


# ── Candidate generation ────────────────────────────────────────────────────
# Maximum weighted Damerau-Levenshtein distance δ from eq. (7). The SymSpell
# index is built with max_dist=2, so this is also the hard ceiling on candidate
# retrieval. Increasing it would require rebuilding the index.
MAX_EDIT_DISTANCE = 2
# Top-K cutoffs for Recall@K (eq. 20).
RECALL_K_VALUES = (1, 5, 10)
# Cap on candidates returned per token, to keep ranking tractable. 10 matches
# the largest Recall@K we report.
MAX_CANDIDATES = 10


# ── Morphological filter (analysis-side re-analysis approximation) ───────────
# The full architecture specifies a morphological synthesiser + reverse-analysis
# verification (eq. 7). That synthesiser is not implemented. As a documented
# approximation, the "DL+FST-filter" candidate mode drops any candidate whose
# reverse morphological analysis fails or whose compatibility coefficient kappa
# falls below KAPPA_MIN. This is analysis-side filtering, not synthesis+reanalysis;
# the distinction is stated explicitly in Section III and the Limitations.
KAPPA_MIN = 0.40  # mirrors kazmorph.mfrn.KAPPA_MIN


# ── Three-way decision rule (eq. 16) ─────────────────────────────────────────
# τ_e: error-probability threshold. A token with p_err < τ_e is kept regardless
#   of candidates. p_err is estimated as 1 - calibrated POS confidence.
TAU_E = 0.15
# τ_c: minimum normalised probability of the leading candidate for CORRECT.
TAU_C = 0.50
# τ_m: minimum margin between top-1 and top-2 candidates for CORRECT.
TAU_M = 0.20
# Grid of thresholds swept for the risk–coverage curve (Fig. 5). The operating
# point (TAU_E, TAU_C, TAU_M) is selected on the validation fold and marked on
# the curve.
TAU_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


# ── Energy / ranking weights (eq. 13) ────────────────────────────────────────
# E(c) = λ_ed·d̃(x,c) + λ_m·Ω_morph(c) - λ_f·log(1+ν_c)
# λ_ctx (contextual log-prob) is zero in this implementation: there is no
# trained contextual ranker, so the contextual term is dropped and disclosed
# as a limitation. The remaining weights are fixed, not tuned on test data.
LAMBDA_ED = 1.0      # edit-distance term
LAMBDA_MORPH = 0.5   # residual morphological uncertainty (1 - kappa)
LAMBDA_FREQ = 0.3    # corpus-frequency bonus (higher freq → lower energy)
LAMBDA_CTX = 0.0     # contextual term — not implemented (no trained ranker)


# ── Stem-expansion candidate channel (dl_freq_morph mode) ────────────────────
# The pure edit-distance index (SymSpell over Apertium lemmas) caps candidate
# recall near 30%: a 1-edit corruption of an inflected surface form
# (e.g. "мектепке" → "мектепкк") cannot be matched because the only form at
# edit-distance 1 is the gold surface form itself, which is excluded as a
# test-fold item (leakage guard, see experiments/candidate_generator.py).
#
# The dl_freq_morph mode adds a second retrieval channel: it decomposes the
# CORRUPTED token into stem + suffixes via ReverseMorphAnalyzer, retrieves
# near-candidates for the STEM, and re-synthesises surface candidates by
# re-attaching the suffixes. This recovers forms whose stem is in the lexicon
# even when the full inflected form is not. It is analysis-side recombination,
# NOT the full morphological synthesiser of eq. (7) — disclosed as such.
# Empirical lift on fold 1: candidate reachability 35.7% → 46.8% (+11.0 pp).
#
# STEM_PENALTY is added to the energy of stem-expanded candidates so they do
# not dominate the direct edit-distance hits (which are more reliable). 0.15
# places a stem candidate roughly one-third of an edit-distance band below a
# direct hit at the same distance — preferred over a direct distance-2 hit,
# below a direct distance-1 hit.
STEM_PENALTY = 0.15
# Cap on stem candidates retrieved per token (keeps the recombination tractable;
# each is O(#suffix-combinations) to expand). 8 mirrors the candidate budget.
STEM_MAX_CANDIDATES = 8


# ── Correction-confidence threshold for the legacy binary corrector ─────────
# Mirrors kazmorph.corrector.correct() default. Used as the "no-REVIEW" baseline
# in the FCR ablation.
LEGACY_CONFIDENCE_THRESHOLD = 0.85


# ── Reproducibility ─────────────────────────────────────────────────────────
# Base encoder used by the restored neural POS/lemma checkpoints.
BASE_ENCODER = "Eraly-ml/KazBERT"
# Calibration temperature loaded from models/neural_pos/calibration.json.
CALIBRATION_TEMPERATURE = 1.75  # updated at runtime if calibration.json present
