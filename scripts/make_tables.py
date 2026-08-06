"""Assemble the paper's results tables (Tables 3–6) from the raw JSON outputs.

Each eval_* script writes results/raw/<name>.json. This script reads them all
and emits Markdown tables ready to paste into the manuscript, plus a combined
results/tables/all_tables.md.

The table numbering matches the paper's revision plan (IHCONCS_wip_plan.md):
  Table 3 — candidate generation (Recall@K, MRR)         ← candidate_recall.json
  Table 4 — final correction (P/R/F0.5/CER/WER/FCR)      ← correction.json
  Table 5 — ablation (FCR: binary vs three-way)          ← fcr.json
  Table 6 — risk–coverage operating points               ← risk_coverage.json

Plus the analyser-coverage table (paper Section IV-C):
  Table (unnumbered) — analyser coverage on UD Kazakh-KTB ← coverage.json

Run:  python scripts/make_tables.py
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402


def _load(name: str) -> dict | None:
    path = os.path.join(config.RAW_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}"


def _ms(x: dict) -> str:
    """mean ± std formatting for a {mean, std} dict, in percent."""
    return f"{_pct(x['mean'])} ± {_pct(x['std'])}"


def table_candidate_recall() -> str:
    d = _load("candidate_recall")
    if d is None:
        return "_(candidate_recall.json not found — run experiments/eval_candidate_recall.py)_\n"
    lines = [
        "**Table 3.** Candidate generation on the controlled error set "
        f"({d['n_folds']}-fold × {d['n_seeds']} seeds).",
        "",
        "| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | Index coverage |",
        "|---|---|---|---|---|---|",
    ]
    label = {"dl": "Damerau–Levenshtein (δ ≤ 2)",
             "dl_freq": "DL + corpus-frequency ranking",
             "dl_fst_filter": "DL + morphological re-analysis filter",
             "dl_freq_morph": "DL + freq + stem recombination (proposed)"}
    for mode in ("dl", "dl_freq", "dl_fst_filter", "dl_freq_morph"):
        if mode not in d["modes"]:
            continue
        a = d["modes"][mode]
        lines.append(
            f"| {label[mode]} | {_ms(a['recall_at_1'])} | {_ms(a['recall_at_5'])} | "
            f"{_ms(a['recall_at_10'])} | {_ms(a['mrr'])} | {_ms(a['index_coverage'])} |"
        )
    lines.append("")
    # Per-category breakdown for the proposed mode (dl_freq_morph). Falls back
    # to dl_fst_filter for older JSONs that don't carry the morph breakdown.
    cat_key = "per_category_dl_freq_morph" if "per_category_dl_freq_morph" in d \
        else "per_category_dl_fst_filter"
    cat_label = "dl_freq_morph" if cat_key == "per_category_dl_freq_morph" else "dl_fst_filter"
    lines.append(
        f"Per-corruption-category Recall@K for the proposed generator "
        f"({cat_label}, fold-averaged, seed 0):"
    )
    lines.append("")
    lines.append("| Corruption category | N | Recall@1 | Recall@5 | Recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|")
    for cat, b in sorted(d[cat_key].items(), key=lambda x: -x[1]["n"]):
        lines.append(
            f"| {cat.replace('_', ' ')} | {b['n']} | {_pct(b['recall_at_1'])} | "
            f"{_pct(b['recall_at_5'])} | {_pct(b['recall_at_10'])} | {_pct(b['mrr'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def table_ablation() -> str:
    """Component ablation — the three ablations Reviewer #2 requested.

    Each candidate-generation mode is a component-level ablation of the full
    proposed generator, restated in the ablation framing the reviewer asked for:
      - "Full proposed"          = dl_freq_morph (freq + stem recombination)
      - "- stem recombination"   = dl_freq      (drops morphology entirely)
      - "- analysis-side filter" = dl_freq_morph vs dl_freq_morph w/o the
                                    validity gate — not separable post-hoc, so
                                    we report dl_fst_filter (filter only, no
                                    recombination) as the analysis-filter row.
    The third reviewer-named ablation ("- morphological lattice encoding") is a
    specified-but-unbuilt component and is reported as such, not fabricated.
    REVIEW-routing is ablated by the binary-vs-three-way row in Table 4.
    """
    d = _load("candidate_recall")
    if d is None:
        return "_(candidate_recall.json not found)_\n"
    lines = [
        "**Table 7.** Component ablation on candidate Recall@K "
        f"({d['n_folds']}-fold × {d['n_seeds']} seeds). Each row removes one "
        "component from the full proposed generator.",
        "",
        "| Configuration | R@1 | R@5 | R@10 | MRR |",
        "|---|---|---|---|---|",
    ]
    rows = [
        ("dl_freq", "− morphology (frequency only)"),
        ("dl_fst_filter", "− stem recombination (analysis-filter only)"),
        ("dl_freq_morph", "**Full proposed (freq + stem recombination)**"),
    ]
    base = d["modes"]["dl_freq_morph"]["recall_at_1"]["mean"]
    for mode, label in rows:
        if mode not in d["modes"]:
            continue
        a = d["modes"][mode]
        delta = a["recall_at_1"]["mean"] - base
        dtag = "—" if abs(delta) < 1e-6 else f"Δ {delta*100:+.1f} pp"
        lines.append(
            f"| {label} ({dtag}) | {_ms(a['recall_at_1'])} | "
            f"{_ms(a['recall_at_5'])} | {_ms(a['recall_at_10'])} | {_ms(a['mrr'])} |"
        )
    lines.append("")
    lines.append(
        "The third reviewer-requested ablation, *− morphological lattice "
        "encoding*, targets a component that is specified in Section III but "
        "not implemented in this report (the RMA returns a single analysis per "
        "token; the multi-path potential reported in Section IV-C motivates "
        "the lattice as future work). It is therefore not included as a "
        "measured row. The *− REVIEW routing* ablation is reported in Table 5 "
        "(binary vs three-way on clean text) and Table 4 (binary vs three-way "
        "on the controlled error set)."
    )
    lines.append("")
    return "\n".join(lines)


def table_correction() -> str:
    d = _load("correction")
    if d is None:
        return "_(correction.json not found — run experiments/eval_correction.py)_\n"
    c = d["calibration"]
    lines = [
        "**Table 4.** Final correction on the controlled error set "
        f"(folds {d['test_folds']} × {d['n_seeds']} seeds; "
        f"thresholds calibrated on fold 0: τ_e={c['tau_e']}, τ_c={c['tau_c']}, τ_m={c['tau_m']}).",
        "",
        "| System | Precision | Recall | F₀.₅ | FCR ↓ | CER ↓ | WER ↓ |",
        "|---|---|---|---|---|---|---|",
    ]
    pretty = {"dl_freq_top1": "DL + frequency, top-1 (no REVIEW)",
              "binary": "Binary corrector (− REVIEW routing)",
              "three_way": "**Proposed neuro-symbolic (KEEP/CORRECT/REVIEW)**"}
    for name in ("dl_freq_top1", "binary", "three_way"):
        if name not in d["configs"]:
            continue
        a = d["configs"][name]
        lines.append(
            f"| {pretty[name]} | {_ms(a['precision'])} | {_ms(a['recall'])} | "
            f"{_ms(a['f05'])} | {_ms(a['fcr'])} | {_ms(a['cer'])} | {_ms(a['wer'])} |"
        )
    lines.append("")
    # 95% CI on F0.5 for the proposed system and the strongest baseline.
    lines.append("95% confidence intervals on F₀.₅ (normal approximation):")
    lines.append("")
    for name in ("dl_freq_top1", "binary", "three_way"):
        if name not in d["configs"]:
            continue
        a = d["configs"][name]
        lo, hi = a["f05_ci95"]
        conf = a["confusion"]
        lines.append(
            f"- {pretty[name]}: F₀.₅ ∈ [{lo:.3f}, {hi:.3f}] "
            f"(TP={conf['tp']}, FP={conf['fp']}, FN={conf['fn']})"
        )
    lines.append("")
    # Paired significance test (Reviewer #2 request).
    sig = d.get("significance")
    if sig and "error" not in sig:
        b = sig.get("bootstrap", {})
        w = sig.get("wilcoxon", {})
        n_pairs = b.get("n_pairs", "?")
        md = b.get("mean_diff", 0.0)
        lo, hi = b.get("ci95", [0.0, 0.0])
        pb = b.get("p_value", float("nan"))
        pw = w.get("p_value", float("nan"))
        lines.append(
            f"Paired significance ({n_pairs} fold×seed pairs), three-way vs binary "
            f"on F₀.₅: Δ = {md:+.4f} (95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]); "
            f"paired bootstrap p = {pb:.4f}, Wilcoxon p = {pw:.4f}."
        )
        lines.append("")
    return "\n".join(lines)


def table_fcr() -> str:
    d = _load("fcr")
    if d is None:
        return "_(fcr.json not found — run experiments/eval_fcr.py)_\n"
    lines = [
        "**Table 5.** False-correction rate on unmodified normative text "
        f"(UD Kazakh-KTB, {d['n_folds']}-fold CV, pooled over {d['n_test_sentences']} test sentences).",
        "",
        "| Configuration | FCR | KEEP | CORRECT | REVIEW |",
        "|---|---|---|---|---|",
    ]
    for mode, label in (("binary", "Binary corrector (− REVIEW)"),
                        ("three_way", "**Three-way (proposed)**")):
        m = d[mode]
        mix = m["decision_mix"]
        lines.append(
            f"| {label} | {_pct(m['fcr'])} ({m['n_changed']}/{m['n_tokens']}) | "
            f"{mix.get('KEEP', 0)} | {mix.get('CORRECT', 0)} | {mix.get('REVIEW', 0)} |"
        )
    lines.append("")
    lines.append("Stratified FCR by token class (binary corrector, pooled):")
    lines.append("")
    lines.append("| Token class | N | FCR |")
    lines.append("|---|---|---|")
    for cls, v in sorted(d["binary"]["by_class"].items(), key=lambda x: -x[1]["total"]):
        lines.append(f"| {cls} | {v['total']} | {_pct(v['fcr'])} ({v['changed']}) |")
    lines.append("")
    return "\n".join(lines)


def table_risk_coverage() -> str:
    d = _load("risk_coverage")
    if d is None:
        return "_(risk_coverage.json not found — run experiments/make_risk_coverage.py)_\n"
    lines = [
        "**Table 6.** Risk–coverage operating points (Fig. 5 source data). "
        f"Tokens evaluated: {d['n_evaluated']}.",
        "",
        "| τ (top-1 prob) | Coverage C(τ) | Risk R(τ) | Auto-decided | Incorrect |",
        "|---|---|---|---|---|",
    ]
    for p in d["curve"]:
        lines.append(
            f"| {p['threshold']:.2f} | {_pct(p['coverage'])} | {_pct(p['risk'])} | "
            f"{p['n_auto']} | {p['n_incorrect']} |"
        )
    lines.append("")
    return "\n".join(lines)


def table_coverage() -> str:
    d = _load("coverage")
    if d is None:
        return "_(coverage.json not found — run experiments/eval_coverage.py)_\n"
    lines = [
        "**Analyser coverage on UD Kazakh-KTB** "
        f"({d['n_tokens']} content tokens, excl. PUNCT/SYM).",
        "",
        f"- Analysis coverage: **{_pct(d['analysis_coverage'])}%** "
        f"({d['n_analysed']}/{d['n_tokens']})",
        f"- Lemma recovery: **{_pct(d['lemma_recovery'])}%** "
        f"({d['n_lemma_correct']}/{d['n_tokens']})",
        f"- Multi-path potential (≥ 2 admissible suffix paths): **{_pct(d['multi_path_fraction'])}%** "
        f"({d['n_multi_path']} tokens)",
        "",
        "Per-UPOS analysis coverage:",
        "",
        "| UPOS | Coverage | N |",
        "|---|---|---|",
    ]
    for upos, b in d["per_upos_coverage"].items():
        lines.append(f"| {upos} | {_pct(b['coverage'])}% | {b['total']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    config.ensure_dirs()
    sections = [
        ("# Results tables — IHCONCS 2026", ""),
        ("## Analyser coverage (Section IV-C)", table_coverage()),
        ("## Table 3 — Candidate generation", table_candidate_recall()),
        ("## Table 4 — Final correction", table_correction()),
        ("## Table 5 — False-correction rate (ablation: − REVIEW)", table_fcr()),
        ("## Table 6 — Risk–coverage", table_risk_coverage()),
        ("## Table 7 — Component ablation", table_ablation()),
    ]
    # Each section is (heading, body). Emit heading, blank line, body, separated
    # by blank lines between sections. Bodies already end in a newline (or are
    # empty, like the title), so we just interleave and strip trailing whitespace.
    parts: list[str] = []
    for heading, body in sections:
        parts.append(heading.rstrip())
        if body.strip():
            parts.append(body.rstrip())
    out = "\n\n".join(parts) + "\n"
    out_path = os.path.join(config.TABLES_DIR, "all_tables.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
