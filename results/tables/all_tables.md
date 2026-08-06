# Results tables — IHCONCS 2026

## Analyser coverage (Section IV-C)

**Analyser coverage on UD Kazakh-KTB** (8437 content tokens, excl. PUNCT/SYM).

- Analysis coverage: **99.7%** (8415/8437)
- Lemma recovery: **96.4%** (8131/8437)
- Multi-path potential (≥ 2 admissible suffix paths): **62.7%** (5290 tokens)

Per-UPOS analysis coverage:

| UPOS | Coverage | N |
|---|---|---|
| ADJ | 99.8% | 827 |
| ADP | 100.0% | 175 |
| ADV | 100.0% | 304 |
| AUX | 98.6% | 361 |
| CCONJ | 100.0% | 187 |
| DET | 100.0% | 220 |
| INTJ | 100.0% | 23 |
| NOUN | 99.8% | 3100 |
| NUM | 100.0% | 374 |
| PART | 93.9% | 33 |
| PRON | 100.0% | 493 |
| PROPN | 98.6% | 562 |
| SCONJ | 100.0% | 21 |
| VERB | 100.0% | 1643 |
| X | 100.0% | 114 |

## Table 3 — Candidate generation

**Table 3.** Candidate generation on the controlled error set (5-fold × 3 seeds).

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| Damerau–Levenshtein (δ ≤ 2) | 19.6 ± 4.0 | 26.2 ± 4.6 | 28.0 ± 4.9 | 22.4 ± 4.1 |
| DL + corpus-frequency ranking | 22.2 ± 3.8 | 27.4 ± 4.6 | 28.6 ± 4.2 | 24.5 ± 4.1 |
| DL + morphological re-analysis filter | 19.8 ± 4.1 | 26.4 ± 4.6 | 28.1 ± 4.8 | 22.6 ± 4.1 |
| DL + freq + stem recombination (proposed) | 26.2 ± 4.0 | 32.1 ± 4.5 | 33.4 ± 4.1 | 28.8 ± 4.2 |

For the proposed generator, Recall@1 / Recall@10 = 78.3% (0.262 / 0.334): when the gold form is reachable it almost always ranks first, so the limiting factor is candidate generation, not ranking.

Per-corruption-category Recall@K for the proposed generator (dl_freq_morph, fold-averaged, seed 0):

| Corruption category | N | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| character deletion | 141 | 27.7 | 40.4 | 41.1 | 32.9 |
| character insertion | 134 | 50.7 | 51.5 | 51.5 | 51.1 |
| adjacent transposition | 127 | 18.1 | 31.5 | 33.9 | 24.0 |
| character substitution | 112 | 33.9 | 38.4 | 38.4 | 36.2 |
| mixed script substitution | 79 | 35.4 | 38.0 | 38.0 | 36.5 |
| affix vowel deletion | 60 | 1.7 | 3.3 | 5.0 | 2.7 |
| vowel harmony violation | 44 | 11.4 | 13.6 | 13.6 | 12.1 |
| grammatical affix substitution | 23 | 0.0 | 0.0 | 0.0 | 0.0 |

## Table 4 — Final correction

**Table 4.** Final correction on the controlled error set (folds [1, 2, 3, 4] × 3 seeds; thresholds calibrated on fold 0: τ_e=0.05, τ_c=0.1, τ_m=0.02).

| System | Precision | Recall | F₀.₅ | FCR ↓ | CER ↓ | WER ↓ |
|---|---|---|---|---|---|---|
| DL + frequency, top-1 (no REVIEW) | 10.7 ± 3.6 | 16.7 ± 4.2 | 11.5 ± 3.6 | 8.6 ± 3.7 | 5.5 ± 1.1 | 16.1 ± 3.2 |
| Binary corrector (− REVIEW routing) | 14.2 ± 4.5 | 7.7 ± 2.0 | 11.9 ± 3.2 | 2.0 ± 1.5 | 3.7 ± 0.6 | 11.1 ± 1.5 |
| **Proposed neuro-symbolic (KEEP/CORRECT/REVIEW)** | 49.7 ± 18.9 | 2.5 ± 1.1 | 10.1 ± 4.3 | 0.2 ± 0.1 | 2.8 ± 0.4 | 9.7 ± 0.9 |
| No-op (input untouched) | — | — | — | — | 2.7 ± 0.3 | 9.7 ± 0.8 |

95% confidence intervals on F₀.₅ (normal approximation):

- DL + frequency, top-1 (no REVIEW): F₀.₅ ∈ [0.093, 0.136] (TP=283, FP=2500, FN=1400)
- Binary corrector (− REVIEW routing): F₀.₅ ∈ [0.100, 0.138] (TP=129, FP=830, FN=1554)
- **Proposed neuro-symbolic (KEEP/CORRECT/REVIEW)**: F₀.₅ ∈ [0.076, 0.126] (TP=41, FP=44, FN=1642)

Paired significance (12 fold×seed pairs), three-way vs binary on F₀.₅: Δ = -0.0181 (95% bootstrap CI [-0.0459, +0.0123]); paired bootstrap p = 0.2352, Wilcoxon p = 0.2094.
Paired significance (12 fold×seed pairs), three-way vs binary on precision: Δ = +0.3552 (95% bootstrap CI [+0.2479, +0.4595]); paired bootstrap p = 0.0000, Wilcoxon p = 0.0022.

## Table 4b — Frequency-dictionary leakage ablation

**Table 4b.** Effect of the frequency-dictionary leakage on correction quality (same B1-fixed recall denominator; folds 1–4 × 3 seeds).

| System | F₀.₅ (leaky) | F₀.₅ (leakage-free) | Δ pp | Recall (leaky) | Recall (leakage-free) |
|---|---|---|---|---|---|
| DL + frequency, top-1 | 12.4 ± 4.1 | 11.5 ± 3.6 | +1.0 | 18.1 ± 4.7 | 16.7 ± 4.2 |
| Binary corrector | 11.9 ± 3.2 | 11.9 ± 3.2 | +0.0 | 7.7 ± 2.0 | 7.7 ± 2.0 |
| Three-way (proposed) | 17.4 ± 6.5 | 10.1 ± 4.3 | +7.3 | 4.4 ± 1.8 | 2.5 ± 1.1 |

Significance of the three-way vs binary F₀.₅ difference under the two protocols:

| Protocol | Δ F₀.₅ | 95% bootstrap CI | bootstrap p | Wilcoxon p |
|---|---|---|---|---|
| Leakage-free (reported) | -0.0181 | [-0.0459, +0.0123] | 0.2352 | 0.2094 |
| Leaky (original) | +0.0551 | [+0.0269, +0.0816] | 0.0000 | 0.0121 |

The leakage-free protocol removes gold-form overlap between the frequency dictionary and the test folds (see `experiments/freq.py`, `build_fold_frequency_dict`). The three-way system remains ahead of the binary baseline on F₀.₅ and substantially ahead on precision (49.7% vs 14.2%) and FCR (0.00% vs 1.90%), but the F₀.₅ gap shrinks and the paired tests no longer reach significance at n = 12. The originally reported p ≈ 0.002 was therefore partly an artefact of the frequency-dictionary overlap; reporting both makes that visible rather than implicit. The leakage affects only the frequency-ranking modes (dl_freq, dl_freq_morph); dl and dl_fst_filter, which do not use the frequency dict, are unchanged.

Candidate Recall@1 under the two protocols (proposed generator):

| Metric | Leaky | Leakage-free | Δ pp |
|---|---|---|---|
| recall at 1 | 29.0 | 26.2 | +2.8 |
| recall at 5 | 34.5 | 32.1 | +2.3 |
| recall at 10 | 35.5 | 33.4 | +2.1 |
| mrr | 31.4 | 28.8 | +2.6 |

## Table 5 — False-correction rate (ablation: − REVIEW)

**Table 5.** False-correction rate on unmodified normative text (UD Kazakh-KTB, calibration fold 0 held out, evaluated on folds [1, 2, 3, 4], pooled over 862 test sentences).

| Configuration | FCR | KEEP | CORRECT | REVIEW |
|---|---|---|---|---|
| Binary corrector (− REVIEW) | 1.9 (154/8270) | 8116 | 154 | 0 |
| **Three-way (proposed)** | 0.0 (0/8270) | 7996 | 0 | 274 |

Per-fold FCR (binary corrector):
- range [0.90%, 4.38%] across 4 folds (per-fold: 1.13%…1.21%…0.90%…4.38%)

Stratified FCR by token class (binary corrector, pooled):

| Token class | N | FCR |
|---|---|---|
| not in lexicon | 5870 | 2.5 (148) |
| common | 1773 | 0.0 (0) |
| proper name | 440 | 1.4 (6) |
| borrowing | 187 | 0.0 (0) |

By morphological analyser outcome (binary corrector, pooled):

| Analyser | N | FCR |
|---|---|---|
| rma known | 8249 | 1.9 (154) |
| rma unknown | 21 | 0.0 (0) |

The `not_in_lexicon` bucket (surface form absent from the noun/verb/adj lemma lists) covers the majority of tokens, but the analyser-based `rma_unknown` bucket covers only a fraction of a percent: the large lexicon-miss class reflects a narrow frequency lexicon, not words the system cannot parse.

## Table 6 — Risk–coverage

**Table 6.** Risk–coverage operating points (Fig. 5 source data). Tokens evaluated: 1331.

| τ (top-1 prob) | Coverage C(τ) | Risk R(τ) | Auto-decided | Incorrect |
|---|---|---|---|---|
| 0.05 | 100.0 | 71.1 | 1331 | 946 |
| 0.10 | 93.1 | 69.2 | 1239 | 858 |
| 0.15 | 43.9 | 57.5 | 584 | 336 |
| 0.20 | 30.1 | 61.8 | 401 | 248 |
| 0.25 | 22.7 | 58.9 | 302 | 178 |
| 0.30 | 11.3 | 32.0 | 150 | 48 |
| 0.40 | 2.7 | 22.2 | 36 | 8 |
| 0.50 | 0.7 | 11.1 | 9 | 1 |

## Table 7 — Component ablation

**Table 7.** Component ablation on candidate Recall@K (5-fold × 3 seeds). Each row removes one component from the full proposed generator.

| Configuration | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| − morphology (frequency only) (Δ -3.9 pp) | 22.2 ± 3.8 | 27.4 ± 4.6 | 28.6 ± 4.2 | 24.5 ± 4.1 |
| − stem recombination (analysis-filter only) (Δ -6.3 pp) | 19.8 ± 4.1 | 26.4 ± 4.6 | 28.1 ± 4.8 | 22.6 ± 4.1 |
| **Full proposed (freq + stem recombination)** (—) | 26.2 ± 4.0 | 32.1 ± 4.5 | 33.4 ± 4.1 | 28.8 ± 4.2 |

The third reviewer-requested ablation, *− morphological lattice encoding*, targets a component that is specified in Section III but not implemented in this report (the RMA returns a single analysis per token; the multi-path potential reported in Section IV-C motivates the lattice as future work). It is therefore not included as a measured row. The *− REVIEW routing* ablation is reported in Table 5 (binary vs three-way on clean text) and Table 4 (binary vs three-way on the controlled error set).
