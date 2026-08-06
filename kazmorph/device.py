"""
Unified device selection for training and inference.

Single source of truth for "which torch device should we use on this machine".
The order is deliberate:

  1. CUDA — fastest, the obvious first choice where available.
  2. MPS  — Apple Silicon GPU. On an M3 Pro 18GB, KazBERT-base fine-tunes in
     ~30-60 min per fold at batch_size 8, vs hours on CPU. This is why MPS
     sits above CPU in the preference order, not below.
  3. CPU  — always available; the safe fallback.

The previous trainers in this repo hard-coded
``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` which silently
fell back to CPU on every Apple Silicon machine, including the development
MacBook this project targets. Replacing that line with ``pick_device()`` is the
one-line change that turns hours-long CPU runs into GPU runs locally.

CPU intra-op thread count is intentionally NOT capped here: the existing
trainers cap it only in the CPU branch (see scripts/train_neural.py:274-276),
and that policy stays where it is. This module only answers the device
question.
"""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Return the best available torch device: cuda > mps > cpu.

    Checks are runtime and re-evaluated on every call, so the result reflects
    the current process state (e.g. if a downstream caller disabled MPS via
    ``torch.backends.mps.is_available = False`` for debugging).

    Determinism: MPS and CUDA use non-deterministic atomic operations in some
    kernels, so two consecutive forward passes on the same input can differ in
    the last bits of the logits. That bit-level noise flips borderline POS
    decisions, which in turn flips which tokens get corrected — so the paper's
    headline TP/FP (and hence precision, recall, F₀.₅) drifts run-to-run by
    ~1-2 percentage points even with corruption seeds fixed. To make results
    reproducible for the reported run, call ``enable_determinism()`` once at
    startup; it seeds the torch RNG and enables deterministic algorithms where
    the installed torch supports them. This is a property of the published
    numbers, not of the device choice itself.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DETERMINISM_SEED = 42


def enable_determinism(seed: int = _DETERMINISM_SEED) -> None:
    """Seed ALL RNGs (Python, NumPy, torch) and enable deterministic algorithms.

    Call once at the start of any experiment whose results must be reproducible
    to the last bit (correction P/R/F₀.₅, significance tests).

    Why seed Python's ``random`` and ``numpy.random`` too, not just torch: the
    KazBERT forward pass itself is deterministic within a process, but some
    downstream library calls (transformers tokenization, sklearn-crfsuite model
    loading) draw from the *global* Python/NumPy RNG during model construction.
    That global RNG is seeded from OS entropy at interpreter startup, so two
    processes produce different model-load sequences even with identical inputs.
    Seeding all three RNGs at startup makes runs reproducible *across processes*
    on the same platform — which is what the paper's reproducibility section
    promises. (Cross-platform bit-identity is still not guaranteed: MPS and CUDA
    atomic ops, and BLAS thread ordering on CPU, can differ between hardware.)

    On backends whose atomic ops are inherently non-deterministic (MPS, some
    CUDA kernels), this pins the forward pass to a fixed RNG stream and enables
    ``use_deterministic_algorithms`` where the installed torch ships them,
    falling back gracefully (warn_only=True) on ops without a deterministic
    implementation rather than aborting.
    """
    import os
    import random
    import numpy as np

    # Hash seed for Python's hash randomization (affects dict ordering in some
    # CPython paths that transformers/tokenizers touch during model load).
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch without warn_only.
        pass
