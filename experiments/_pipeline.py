"""Shared pipeline factory and corpus loader for all experiment scripts.

Centralising construction here guarantees every script loads the restored
KazBERT POS/lemma checkpoints with the same arguments, and reads the gold
pool through the leakage-free `corpus.load_pool()` helper.
"""

from __future__ import annotations

import os
import sys

# Make `kazmorph` and the project root importable when scripts are run directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Heavy ML dependencies (torch, transformers, numpy, matplotlib, scikit-learn)
# live in the project venv — see bin/py and requirements.txt. On Linux the venv
# is created with `uv venv` (CUDA torch from the cu128 index). There is no
# need to splice another environment's site-packages onto sys.path here; every
# experiment script is launched through bin/py, which selects that venv.

from kazmorph.pipeline import Pipeline  # noqa: E402
from kazmorph.corpus import load_pool, stratified_kfold  # noqa: E402

import config  # noqa: E402


def build_pipeline(*, routing: bool = False, mfrn_scoring: bool = False) -> Pipeline:
    """Return a Pipeline wired to the restored models.

    `routing=False` (default) is the linear cascade RMA→Neural→Lemma→MFRN,
    which is the production path and the one used for all reported numbers.
    `routing=True` enables the error-class-conditioned fusion (paper §"routing"),
    used only for the routing ablation if reported.
    """
    return Pipeline(
        crf_model_path=config.CRF_MODEL_PATH,
        neural_model_path=config.NEURAL_POS_PATH,
        lemma_model_path=config.NEURAL_LEMMA_PATH,
        routing=routing,
        mfrn_scoring=mfrn_scoring,
    )


def load_gold_pool():
    """Return the deduplicated gold sentence pool (silver files excluded)."""
    return load_pool()


def make_folds(n_folds: int | None = None, seed: int | None = None):
    """Return list[(train_sents, dev_sents)] stratified k-fold splits.

    Corruption is applied INSIDE each fold after splitting (see corrupt.py),
    so a normative sentence can never leak from train into a corrupted test
    pair. This is the invariant the paper's reproducibility section relies on.
    """
    pool = load_gold_pool()
    return stratified_kfold(
        pool,
        k=config.N_FOLDS if n_folds is None else n_folds,
        seed=config.CV_SEED if seed is None else seed,
    )
