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

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | Index coverage |
|---|---|---|---|---|---|
| Damerau–Levenshtein (δ ≤ 2) | 21.1 ± 3.3 | 27.6 ± 4.5 | 29.0 ± 4.6 | 23.8 ± 3.4 | 29.0 ± 4.6 |
| DL + corpus-frequency ranking | 24.2 ± 3.2 | 29.2 ± 4.6 | 30.4 ± 4.3 | 26.4 ± 3.7 | 30.4 ± 4.3 |
| DL + morphological re-analysis filter | 21.2 ± 3.3 | 27.7 ± 4.4 | 29.1 ± 4.6 | 23.9 ± 3.4 | 29.1 ± 4.6 |
| DL + freq + stem recombination (proposed) | 28.5 ± 3.2 | 33.8 ± 4.2 | 34.9 ± 4.2 | 30.9 ± 3.5 | 34.9 ± 4.2 |

Per-corruption-category Recall@K for the proposed generator (dl_freq_morph, fold-averaged, seed 0):

| Corruption category | N | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| character deletion | 133 | 33.8 | 48.1 | 48.1 | 39.8 |
| character insertion | 133 | 52.6 | 52.6 | 52.6 | 52.6 |
| character substitution | 121 | 33.9 | 37.2 | 37.2 | 35.4 |
| adjacent transposition | 120 | 16.7 | 30.8 | 32.5 | 22.0 |
| mixed script substitution | 79 | 31.6 | 31.6 | 31.6 | 31.6 |
| affix vowel deletion | 59 | 0.0 | 3.4 | 5.1 | 1.9 |
| vowel harmony violation | 47 | 14.9 | 14.9 | 14.9 | 14.9 |
| grammatical affix substitution | 22 | 0.0 | 0.0 | 0.0 | 0.0 |

## Table 4 — Final correction

**Table 4.** Final correction on the controlled error set (folds [1, 2, 3, 4] × 3 seeds; thresholds calibrated on fold 0: τ_e=0.05, τ_c=0.15, τ_m=0.0).

| System | Precision | Recall | F₀.₅ | FCR ↓ | CER ↓ | WER ↓ |
|---|---|---|---|---|---|---|
| DL + frequency, top-1 (no REVIEW) | 11.2 ± 3.6 | 24.9 ± 5.8 | 12.5 ± 3.8 | 8.5 ± 3.7 | 5.4 ± 1.1 | 16.0 ± 3.2 |
| Binary corrector (− REVIEW routing) | 14.4 ± 4.3 | 9.7 ± 2.3 | 12.9 ± 3.3 | 2.0 ± 1.5 | 3.7 ± 0.5 | 11.1 ± 1.4 |
| **Proposed neuro-symbolic (KEEP/CORRECT/REVIEW)** | 68.6 ± 16.8 | 4.5 ± 1.9 | 17.5 ± 6.5 | 0.1 ± 0.1 | 2.7 ± 0.3 | 9.5 ± 0.7 |

95% confidence intervals on F₀.₅ (normal approximation):

- DL + frequency, top-1 (no REVIEW): F₀.₅ ∈ [0.102, 0.148] (TP=290, FP=2450, FN=872)
- Binary corrector (− REVIEW routing): F₀.₅ ∈ [0.109, 0.148] (TP=128, FP=816, FN=1193)
- **Proposed neuro-symbolic (KEEP/CORRECT/REVIEW)**: F₀.₅ ∈ [0.137, 0.213] (TP=75, FP=33, FN=1593)

Paired significance (12 fold×seed pairs), three-way vs binary on F₀.₅: Δ = +0.0466 (95% bootstrap CI [+0.0171, +0.0753]); paired bootstrap p = 0.0020, Wilcoxon p = 0.0186.

## Table 5 — False-correction rate (ablation: − REVIEW)

**Table 5.** False-correction rate on unmodified normative text (UD Kazakh-KTB, 5-fold CV, pooled over 1078 test sentences).

| Configuration | FCR | KEEP | CORRECT | REVIEW |
|---|---|---|---|---|
| Binary corrector (− REVIEW) | 1.7 (175/10536) | 10361 | 175 | 0 |
| **Three-way (proposed)** | 0.0 (0/10536) | 10228 | 0 | 308 |

Stratified FCR by token class (binary corrector, pooled):

| Token class | N | FCR |
|---|---|---|
| oov | 7539 | 2.2 (169) |
| common | 2215 | 0.0 (0) |
| proper_name | 562 | 1.1 (6) |
| borrowing | 220 | 0.0 (0) |

## Table 6 — Risk–coverage

**Table 6.** Risk–coverage operating points (Fig. 5 source data). Tokens evaluated: 1331.

| τ (top-1 prob) | Coverage C(τ) | Risk R(τ) | Auto-decided | Incorrect |
|---|---|---|---|---|
| 0.05 | 100.0 | 69.0 | 1331 | 918 |
| 0.10 | 93.8 | 67.1 | 1248 | 838 |
| 0.15 | 45.1 | 54.7 | 600 | 328 |
| 0.20 | 31.6 | 58.6 | 420 | 246 |
| 0.25 | 24.0 | 56.2 | 320 | 180 |
| 0.30 | 13.1 | 30.5 | 174 | 53 |
| 0.40 | 5.3 | 12.9 | 70 | 9 |
| 0.50 | 1.0 | 15.4 | 13 | 2 |

## Table 7 — Component ablation

**Table 7.** Component ablation on candidate Recall@K (5-fold × 3 seeds). Each row removes one component from the full proposed generator.

| Configuration | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| − morphology (frequency only) (Δ -4.3 pp) | 24.2 ± 3.2 | 29.2 ± 4.6 | 30.4 ± 4.3 | 26.4 ± 3.7 |
| − stem recombination (analysis-filter only) (Δ -7.3 pp) | 21.2 ± 3.3 | 27.7 ± 4.4 | 29.1 ± 4.6 | 23.9 ± 3.4 |
| **Full proposed (freq + stem recombination)** (—) | 28.5 ± 3.2 | 33.8 ± 4.2 | 34.9 ± 4.2 | 30.9 ± 3.5 |

The third reviewer-requested ablation, *− morphological lattice encoding*, targets a component that is specified in Section III but not implemented in this report (the RMA returns a single analysis per token; the multi-path potential reported in Section IV-C motivates the lattice as future work). It is therefore not included as a measured row. The *− REVIEW routing* ablation is reported in Table 5 (binary vs three-way on clean text) and Table 4 (binary vs three-way on the controlled error set).
