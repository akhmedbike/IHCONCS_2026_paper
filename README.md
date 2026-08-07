# Neuro-Symbolic Correction of Kazakh Word Forms (IHCONCS 2026)

Experimental code and reproducible results for the IHCONCS 2026 revision of
*"Neuro-Symbolic Correction of Kazakh Word Forms through Morphological Analysis
and Contextual Disambiguation."*

This repository accompanies the **work-in-progress / practical experience
report** version of the paper. It implements the five measurable quantities
the revision adds to Section IV, on top of the vendored `kazmorph` morphological
pipeline. No neural networks are trained here — the KazBERT POS and lemma
checkpoints are restored from the `kazmorph` project and used for inference
only.

## What this repo measures

| # | Quantity | Script | Headline result |
|---|---|---|---|
| N1 ⭐ | False-correction rate on clean text | `experiments/eval_fcr.py` | binary 1.7% → three-way 0.0% |
| N2 | Morphological analyser coverage | `experiments/eval_coverage.py` | 99.7% analysis, 96.4% lemma, 62.7% multi-path |
| N3 | Candidate Recall@1/5/10 | `experiments/eval_candidate_recall.py` | dl_freq_morph best (28.5 / 33.8 / 34.9%) |
| N4 | End-to-end P / R / F₀.₅ / CER / WER | `experiments/eval_correction.py` | proposed F₀.₅ = 17.5%, FCR 0.1% |
| N5 | Risk–coverage curve (Fig. 5) | `experiments/make_risk_coverage.py` | see Table 6 |

Significance (paired bootstrap, 10 000 resamples, three-way vs binary on F₀.₅):
Δ = +0.047, 95% CI [+0.017, +0.075], **p = 0.002** (Wilcoxon p = 0.019).

## Quick start

Download models from Zenodo (https://doi.org/10.5281/zenodo.21820946)

```bash
# 1. Create a Python 3.14 venv (uv is the supported installer). The venv MUST
#    live on a symlink-capable filesystem — the project tree itself sits on
#    exFAT, which cannot host a venv. ~/Dev/paper3-venv/.venv is the default
#    bin/py looks for; override with the PAPER3_VENV env var.
uv venv --python 3.14 "$HOME/Dev/paper3-venv/.venv"

# 2. Install torch from the CUDA wheel index (cu128), then the rest.
uv pip install --python "$HOME/Dev/paper3-venv/.venv/bin/python" \
    --index https://download.pytorch.org/whl/cu128 torch>=2.10.0
uv pip install --python "$HOME/Dev/paper3-venv/.venv/bin/python" -r requirements.txt

# 3. Run every experiment in dependency order. bin/py selects the venv above.
./bin/py experiments/run_all.py
```

This runs every experiment in dependency order and assembles
`results/tables/all_tables.md`. To run a single step:

```bash
./bin/py -m experiments.eval_fcr
./bin/py -m experiments.eval_correction
./bin/py -m scripts.make_tables
```

## Reference hardware

The reported numbers were produced on a Linux machine with an NVIDIA GPU
(CUDA inference; `kazmorph/device.py` picks `cuda > mps > cpu` automatically):

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 9 8945HS, 16 threads @ 5.26 GHz |
| GPU | NVIDIA GeForce RTX 4070 Max-Q (Ada, sm_89), discrete |
| iGPU | AMD Radeon 780M (unused by this code) |
| RAM | 32 GiB (7.8 GiB resident during runs) |
| OS | CachyOS Linux 7.1.5, Wayland / niri 26.04 |
| Torch | 2.10.0 (cu128 wheels, bundled CUDA 12.8 runtime) |

The code is hardware-agnostic and also runs on Apple Silicon (MPS) or CPU;
only wall time changes. End-to-end (`run_all.py`) is dominated by KazBERT
inference for N1 and N4; on the reference GPU the full sweep completes in a
few minutes.

## Four candidate-generation modes

| Mode | Retrieval | Ranking | Role in the paper |
|---|---|---|---|
| `dl` | SymSpell (δ ≤ 2) | (distance, alpha) | baseline [1], [2] |
| `dl_freq` | SymSpell | (distance, −frequency, alpha) | data-driven baseline [3] |
| `dl_fst_filter` | SymSpell + κ-gate | (distance, alpha) | analysis-side re-analysis approximation |
| `dl_freq_morph` | SymSpell + κ-gate + stem recombination | (distance, −frequency, morphology) | **proposed** |

The `dl_freq_morph` mode adds a second retrieval channel: the corrupted token
is decomposed into stem + suffixes via `ReverseMorphAnalyzer`, near-candidates
for the **stem** are retrieved, and surface forms are re-synthesised by
re-attaching the suffixes. This lifts candidate reachability from 35.7% to
46.8% on fold 1. It is analysis-side recombination, **not** the full
morphological synthesiser of eq. (7); the distinction is disclosed in the
Limitations.

## Repository layout

```
paper3/
  config.py                  # paths, thresholds, seeds — single source of truth
  kazmorph/                  # vendored morphological pipeline (from kazmorph, repo - https://github.com/akhmedbike/kazmorph)
  data/                      # UD Kazakh-KTB + silver corpora
  models/                    # restored KazBERT POS + lemma checkpoints
  experiments/
    _pipeline.py             # Pipeline factory + leakage-free gold loader
    correction_metrics.py    # F0.5, WER, CER, MRR, Recall@K, risk-coverage
    keyboards.py             # Kazakh keyboard neighbours + mixed-script map
    freq.py                  # corpus-frequency dict (for DL+freq baseline)
    corrupt.py               # 9 controlled-error operators (Table 2)
    candidate_generator.py   # dl / dl_freq / dl_fst_filter / dl_freq_morph modes
    corrector_3way.py        # KEEP / CORRECT / REVIEW decision rule (eq. 16)
    eval_coverage.py         # N2
    eval_fcr.py              # N1 ⭐
    eval_candidate_recall.py # N3
    eval_correction.py       # N4 (calibrates thresholds on fold 0)
    make_risk_coverage.py    # N5 + Fig. 5
    significance.py          # paired bootstrap + Wilcoxon (reviewer request)
    run_all.py               # orchestrator
  scripts/
    make_tables.py           # assembles Tables 3–6 from results/raw/*.json
    restore_models.sh        # restore KazBERT weights (not committed) from kazmorph
  results/
    raw/                     # per-experiment JSON (committed snapshot, see below)
    tables/                  # all_tables.md — paste into the manuscript
    figures/                 # fig5_risk_coverage.png
```

### About the committed `results/` snapshot

`results/raw/`, `results/tables/`, and `results/figures/` **are committed on
purpose** — they are the evidence behind every number in the paper and in
`REVISION_PROPOSAL.md` (108 KB total; the per-corruption intermediate files in
`results/controlled_errors/` are *not* committed, they regenerate in seconds).
This is the "trusted snapshot" convention for evaluation repos: a reader or
reviewer can open `results/tables/all_tables.md` and see the headline numbers
without installing CUDA, downloading ~840 MB of model weights, and running the
sweep.

**The committed files are fully regenerable.** The source of truth is the code,
not the snapshot:

```bash
./bin/py experiments/run_all.py        # regenerates every file in results/
git status                             # if results/ is clean, snapshot == code
```

If you change the code, re-run `run_all.py` and commit the updated JSON
alongside the code change — do not edit `results/*.json` by hand. The integrity
notes below explain why every value is traceable to a script.

## Integrity notes (read before citing any number)

- Every number in `results/` is produced by a script reading real data on disk.
  No hand-entered values.
- Thresholds (τ_e, τ_c, τ_m) are calibrated on **fold 0 only** and locked before
  evaluating folds 1–4. The test folds are never used for threshold selection,
  frequency estimation, or early stopping.
- Corruption is applied **inside each fold after splitting**, so a normative
  sentence cannot leak from training into a corrupted test pair.
- The paper originally claimed 10-fold CV and 5 random seeds; the shipped harness
  implements **5-fold** and we run **3 seeds**. This reduction is disclosed in
  Section III and reflected here in `config.N_FOLDS` / `config.N_CORRUPTION_SEEDS`.

## What is NOT implemented (disclosed in the paper's Limitations)

The paper's full architecture specifies four components this repo does not yet
implement. They are described in Section III as design specifications and listed
explicitly in the Discussion / Limitations:

- **Morphological lattice** — the RMA returns a single analysis per token; the
  multi-path potential (N2, 62.7% of content tokens admit ≥2 suffix paths) is
  reported as motivation for the future lattice.
- **Morphological synthesiser + reverse-analysis verification** — the
  `dl_freq_morph` candidate mode is an *analysis-side* approximation (stem
  recombination + κ-gated re-parsing), not the full synthesis→reanalysis cycle
  of eq. (7). A true synthesiser would close the residual Recall gap on the
  morphological corruption operators (affix-vowel deletion, harmony violation),
  which the SymSpell + stem channel still largely misses (Table 3 per-category).
- **Multi-granular encoder with learned gating** — the contextual term λ_ctx
  is zero; candidate ranking uses edit-distance + morphology + frequency only.
  A parallel implementation of the gated encoder exists in the group's
  `paper1` (CAGF) repository and is cited as a feasibility reference.
- **Joint training objective** — the POS and lemma checkpoints were trained
  independently in `kazmorph`; no joint loss is optimised here.

## Provenance

- `kazmorph/` — vendored copy of the `kazmorph` package (commit as of
  2026-08-04). See `kazmorph/WORKLOG.md` for its development history.
- `data/` — UD Kazakh-KTB (gold) + Apertium-derived silver corpora, copied
  verbatim from `kazmorph/data`.
- `models/neural_pos` — KazBERT (`Eraly-ml/KazBERT`) fine-tuned for POS tagging,
  5-fold CV F1 ≈ 0.885. Restored from the
  `neural_pos.kazbert_nosilver_backup` checkpoint.
- `models/neural_lemma` — KazBERT fine-tuned for lemma-prefix prediction.

## Funding

Grant AP23487753, Science Committee of the Ministry of Science and Higher
Education of the Republic of Kazakhstan.
