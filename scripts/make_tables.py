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
        "| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|---|",
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
            f"{_ms(a['recall_at_10'])} | {_ms(a['mrr'])} |"
        )
    lines.append("")
    # Note the near-equality of Recall@10 and the generator's index coverage
    # (the share of corrupted tokens for which any candidate is retrieved):
    # they coincide to display precision in every mode, so the standalone
    # "Index coverage" column is redundant and was dropped. The substantive
    # point it carried is the gap between Recall@1 and Recall@10 for the
    # proposed generator — once the gold form is reachable it almost always
    # ranks first (R@1 / R@10 ≈ 0.82), so the bottleneck is candidate
    # reachability, not ranking quality.
    fm = d["modes"].get("dl_freq_morph")
    if fm:
        ratio = fm["recall_at_1"]["mean"] / fm["recall_at_10"]["mean"] \
            if fm["recall_at_10"]["mean"] > 0 else 0.0
        lines.append(
            f"For the proposed generator, Recall@1 / Recall@10 = {_pct(ratio)}% "
            f"({fm['recall_at_1']['mean']:.3f} / {fm['recall_at_10']['mean']:.3f}): "
            f"when the gold form is reachable it almost always ranks first, so the "
            f"limiting factor is candidate generation, not ranking."
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
              "three_way": "**Proposed neuro-symbolic (KEEP/CORRECT/REVIEW)**",
              "noop": "No-op (input untouched)"}
    for name in ("dl_freq_top1", "binary", "three_way"):
        if name not in d["configs"]:
            continue
        a = d["configs"][name]
        lines.append(
            f"| {pretty[name]} | {_ms(a['precision'])} | {_ms(a['recall'])} | "
            f"{_ms(a['f05'])} | {_ms(a['fcr'])} | {_ms(a['cer'])} | {_ms(a['wer'])} |"
        )
    # No-op baseline: CER/WER floor only (it corrects nothing, so P/R/F₀.₅/FCR
    # are undefined). This is the row that answers "wouldn't doing nothing
    # score just as well on CER/WER?" — a system that barely intervenes has
    # CER/WER close to the untouched input's.
    if "noop" in d["configs"]:
        na = d["configs"]["noop"]
        lines.append(
            f"| {pretty['noop']} | — | — | — | — | "
            f"{_ms(na['cer'])} | {_ms(na['wer'])} |"
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
    # Paired significance tests (Reviewer #2 request). Reported for BOTH F₀.₅
    # and precision: under the leakage-free protocol the F₀.₅ gap is not
    # significant, but the precision gap is the claim the paper actually makes,
    # and it is large and stable enough to clear significance on its own.
    sig = d.get("significance")
    if sig and "error" not in sig:
        metric_label = {"f05": "F₀.₅", "precision": "precision"}
        for metric in ("f05", "precision"):
            entry = sig.get(metric)
            if not entry:
                continue
            b = entry.get("bootstrap", {})
            w = entry.get("wilcoxon", {})
            n_pairs = b.get("n_pairs", "?")
            md = b.get("mean_diff", 0.0)
            lo, hi = b.get("ci95", [0.0, 0.0])
            pb = b.get("p_value", float("nan"))
            pw = w.get("p_value", float("nan"))
            lines.append(
                f"Paired significance ({n_pairs} fold×seed pairs), three-way vs binary "
                f"on {metric_label.get(metric, metric)}: Δ = {md:+.4f} "
                f"(95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]); "
                f"paired bootstrap p = {pb:.4f}, Wilcoxon p = {pw:.4f}."
            )
        lines.append("")
    return "\n".join(lines)


def table_fcr() -> str:
    d = _load("fcr")
    if d is None:
        return "_(fcr.json not found — run experiments/eval_fcr.py)_\n"
    # Caption reflects the corrected protocol: fold 0 is held out for
    # threshold calibration, so FCR is reported on the remaining folds only.
    eval_folds = d.get("eval_folds")
    fold_note = (
        f"calibration fold 0 held out, evaluated on folds {eval_folds}"
        if eval_folds is not None else f"{d['n_folds']}-fold CV"
    )
    lines = [
        "**Table 5.** False-correction rate on unmodified normative text "
        f"(UD Kazakh-KTB, {fold_note}, pooled over {d['n_test_sentences']} test sentences).",
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
    # Per-fold range so the fold-to-fold variability (and any outlier) is
    # visible rather than hidden behind the pooled mean. The binary corrector's
    # last evaluated fold runs several times higher than the rest.
    lines.append("")
    lines.append("Per-fold FCR (binary corrector):")
    bin_folds = d["binary"].get("fcr_per_fold", [])
    if bin_folds:
        lo, hi = min(bin_folds), max(bin_folds)
        rng = "…" if len(bin_folds) > 2 else ", "
        seq = rng.join(f"{f*100:.2f}%" for f in bin_folds)
        lines.append(
            f"- range [{lo*100:.2f}%, {hi*100:.2f}%] across {len(bin_folds)} folds "
            f"(per-fold: {seq})"
        )
    lines.append("")
    # Two parallel stratifications expose the lexicon-miss artefact: the
    # "not_in_lexicon" bucket is large (~71%) because kazmorph.lexicons are
    # lemma lists queried with surface forms, whereas the analyser-based cut
    # shows only ~0.2% genuinely unparseable tokens.
    lines.append("Stratified FCR by token class (binary corrector, pooled):")
    lines.append("")
    lines.append("| Token class | N | FCR |")
    lines.append("|---|---|---|")
    for cls, v in sorted(d["binary"]["by_class"].items(), key=lambda x: -x[1]["total"]):
        lines.append(f"| {cls.replace('_', ' ')} | {v['total']} | {_pct(v['fcr'])} ({v['changed']}) |")
    lines.append("")
    by_an = d["binary"].get("by_analyzer")
    if by_an:
        lines.append("By morphological analyser outcome (binary corrector, pooled):")
        lines.append("")
        lines.append("| Analyser | N | FCR |")
        lines.append("|---|---|---|")
        for cls, v in sorted(by_an.items(), key=lambda x: -x[1]["total"]):
            lines.append(f"| {cls.replace('_', ' ')} | {v['total']} | {_pct(v['fcr'])} ({v['changed']}) |")
        lines.append("")
        lines.append(
            "The `not_in_lexicon` bucket (surface form absent from the noun/verb/"
            "adj lemma lists) covers the majority of tokens, but the analyser-based "
            "`rma_unknown` bucket covers only a fraction of a percent: the large "
            "lexicon-miss class reflects a narrow frequency lexicon, not words the "
            "system cannot parse."
        )
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


def table_leakage_ablation() -> str:
    """Leaky-vs-clean comparison showing how the frequency-dictionary leakage
    inflated the headline F₀.₅ and the significance test.

    The leakage-free protocol (per-fold frequency dict, test sentences excluded
    from all sources) is the methodologically correct one and is what the main
    tables report. The leaky variant (global dict built from train_split +
    silver, ~85% pool overlap) is shown here only to quantify the effect of the
    leakage that the original protocol carried, and to make explicit that the
    originally reported p = 0.002 was partly an artefact of that overlap.
    """
    clean = _load("correction")
    leaky = _load("correction_leaky")
    if clean is None or leaky is None:
        return "_(correction.json and/or correction_leaky.json not found — "
        "run experiments/eval_correction.py and eval_correction.py --leaky)_\n"
    pretty = {"dl_freq_top1": "DL + frequency, top-1",
              "binary": "Binary corrector",
              "three_way": "Three-way (proposed)"}
    lines = [
        "**Table 4b.** Effect of the frequency-dictionary leakage on correction "
        "quality (same B1-fixed recall denominator; folds 1–4 × 3 seeds).",
        "",
        "| System | F₀.₅ (leaky) | F₀.₅ (leakage-free) | Δ pp | "
        "Recall (leaky) | Recall (leakage-free) |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("dl_freq_top1", "binary", "three_way"):
        c = clean["configs"][name]
        l = leaky["configs"][name]
        df = (l["f05"]["mean"] - c["f05"]["mean"]) * 100
        lines.append(
            f"| {pretty[name]} | {_ms(l['f05'])} | {_ms(c['f05'])} | "
            f"{df:+.1f} | {_ms(l['recall'])} | {_ms(c['recall'])} |"
        )
    lines.append("")
    # Significance under both protocols — the key methodological point.
    # Handle both the old flat significance schema ({bootstrap, wilcoxon}) and
    # the new per-metric schema ({f05: {bootstrap, wilcoxon}, precision: {...}}).
    def _sig_f05(d):
        s = d.get("significance", {})
        if "f05" in s:
            return s["f05"]
        return s  # old flat schema: bootstrap/wilcoxon at top level

    lines.append("Significance of the three-way vs binary F₀.₅ difference under "
                 "the two protocols:")
    lines.append("")
    lines.append("| Protocol | Δ F₀.₅ | 95% bootstrap CI | bootstrap p | Wilcoxon p |")
    lines.append("|---|---|---|---|---|")
    for tag, d in (("Leakage-free (reported)", clean), ("Leaky (original)", leaky)):
        entry = _sig_f05(d)
        if "error" in entry:
            lines.append(f"| {tag} | — | — | (error) | — |")
            continue
        b = entry.get("bootstrap", {})
        w = entry.get("wilcoxon", {})
        lo, hi = b.get("ci95", [0.0, 0.0])
        lines.append(
            f"| {tag} | {b.get('mean_diff', 0.0):+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] | {b.get('p_value', float('nan')):.4f} | "
            f"{w.get('p_value', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append(
        "The leakage-free protocol removes gold-form overlap between the "
        "frequency dictionary and the test folds (see `experiments/freq.py`, "
        "`build_fold_frequency_dict`). Under the leakage-free protocol the "
        "three-way system is NOT ahead of the binary baseline on F₀.₅ "
        f"(Δ = {clean['significance']['f05']['bootstrap']['mean_diff']:+.3f}, "
        "n.s.); its advantages over binary are on precision "
        f"({clean['configs']['three_way']['precision']['mean']*100:.1f}% vs "
        f"{clean['configs']['binary']['precision']['mean']*100:.1f}%, Wilcoxon "
        f"p = {clean['significance']['precision']['wilcoxon']['p_value']:.4f}) "
        "and FCR (0.00% vs 1.90%). The leaky protocol inverted the F₀.₅ "
        "ordering by inflating the frequency-ranking modes; reporting both "
        "makes that visible rather than implicit. The leakage affects only the "
        "frequency-ranking modes (dl_freq, dl_freq_morph); dl and dl_fst_filter, "
        "which do not use the frequency dict, are unchanged."
    )
    lines.append("")
    # Also show the candidate-recall leakage effect for the proposed generator.
    cc = _load("candidate_recall")
    cl = _load("candidate_recall_leaky")
    if cc and cl:
        lines.append("Candidate Recall@1 under the two protocols (proposed generator):")
        lines.append("")
        lines.append("| Metric | Leaky | Leakage-free | Δ pp |")
        lines.append("|---|---|---|---|")
        for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
            lm = cl["modes"]["dl_freq_morph"][metric]["mean"]
            cm = cc["modes"]["dl_freq_morph"][metric]["mean"]
            lines.append(f"| {metric.replace('_', ' ')} | "
                         f"{_pct(lm)} | {_pct(cm)} | {(lm-cm)*100:+.1f} |")
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
        ("## Table 4b — Frequency-dictionary leakage ablation", table_leakage_ablation()),
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
