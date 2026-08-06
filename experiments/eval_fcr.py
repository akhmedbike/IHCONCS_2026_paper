"""N1 — False-correction rate on unmodified normative text. ⭐

This is the single most important number in the paper, and the cheapest to
obtain: run the corrector on clean gold sentences and count how many tokens
it changed. No error corpus is needed.

Why it matters: every prior Kazakh correction paper ([1]–[3]) reports only
the accuracy of fixing injected errors. None reports how often the system
falsely "corrects" an already-correct token — yet the paper's entire argument
for the REVIEW mechanism rests on the claim that false corrections are common
and costly. This evaluation produces the first such measurement for Kazakh
that the authors are aware of.

We compare two configurations on the SAME clean text:

  * "binary"   — the legacy corrector (low-conf/UNK → top suggestion, no REVIEW).
                 This is the "− REVIEW routing" ablation: every flagged token
                 is changed automatically.

  * "three_way" — the KEEP/CORRECT/REVIEW rule. REVIEW retains the original
                 token, so only CORRECT decisions contribute to FCR.

For each configuration we report:
  - FCR overall (paper eq. 22).
  - Decision mix (KEEP / CORRECT / REVIEW counts).
  - Stratified FCR by token class: proper names (PROPN), borrowings, OOV
    (outside the lexicon), and rare forms (lowest corpus-frequency quartile).
    The stratification shows WHERE the false corrections concentrate — the
    paper's narrative is that they hit rare words and names, exactly the
    tokens a purely accuracy-tuned system sacrifices.

Run:  python experiments/eval_fcr.py
Writes: results/raw/fcr.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from experiments._pipeline import build_pipeline, make_folds  # noqa: E402
from experiments.corrector_3way import (  # noqa: E402
    correct_token_3way, correct_token_binary,
)
from experiments.freq import get_frequency_dict, frequency_of  # noqa: E402
from kazmorph.lexicons import is_known_noun, is_known_verb, is_known_adj, is_borrowing  # noqa: E402
from kazmorph.rma import ReverseMorphAnalyzer  # noqa: E402


# Module-level RMA for the analyzer-based token class. Read-only after init,
# safe to share across folds. Constructed lazily so `--help` etc. stay cheap.
_RMA: ReverseMorphAnalyzer | None = None


def _get_rma() -> ReverseMorphAnalyzer:
    global _RMA
    if _RMA is None:
        _RMA = ReverseMorphAnalyzer()
    return _RMA


def _token_class(form: str, upos: str, freq: dict[str, int]) -> str:
    """Categorise a token for FCR stratification (lexicon-miss cut).

    Order matters: the first matching class wins. PROPN is checked first
    because a borrowing that is also a proper name (e.g. a Russian toponym)
    is most usefully counted as a name for the FCR narrative.

    Note on the "not_in_lexicon" bucket (formerly "oov"): kazmorph.lexicons
    holds *lemma* lists (KNOWN_NOUNS/VERBS/ADJ, ~22k entries total generated
    from Apertium-kaz), but this function queries them with the surface form.
    For an agglutinative language most inflected surface tokens therefore miss,
    which is why ~71% of tokens land here despite analyser coverage of 99.7%.
    The bucket is renamed to reflect what it actually measures — "not literally
    present in the noun/verb/adj lemma list as this surface form" — not "the
    word is unknown to the language". The analyser-based companion cut
    (_analyzer_class) reports a structurally meaningful "could not parse" rate.
    """
    if upos == "PROPN":
        return "proper_name"
    if is_borrowing(form):
        return "borrowing"
    lower = form.lower()
    if not (is_known_noun(lower) or is_known_verb(lower) or is_known_adj(lower)):
        return "not_in_lexicon"
    return "common"


def _analyzer_class(form: str) -> str:
    """Categorise a token by whether the morphological analyser can parse it.

    This is the second, structurally meaningful cut for the FCR stratification.
    A token is "rma_unknown" iff ReverseMorphAnalyzer returns UNK with no
    suffixes — i.e. no stem and no morphemes were recovered — and "rma_known"
    otherwise. On the Kazakh-KTB pool this is ~0.2% unknown vs ~72% in the
    lexicon-miss bucket, which is the contrast the paper needs to state: the
    frequency lexicon is narrow, but the analyser covers the language.
    """
    res = _get_rma().analyze(form.lower())
    return "rma_unknown" if (res.pos == "UNK" and not res.suffixes) else "rma_known"


def _is_rare(form: str, freq: dict[str, int], rare_threshold: int) -> bool:
    """A token is 'rare' if its corpus frequency is below the threshold."""
    return frequency_of(form, freq) < rare_threshold


def _rare_threshold(freq: dict[str, int]) -> int:
    """Frequency below which a token falls in the bottom quartile.

    Computed over the union of (gold pool forms ∩ frequency dict). This is a
    data-driven split, not a magic number — we report the threshold value in
    the output so the paper can cite it.
    """
    values = sorted(freq.values())
    if not values:
        return 1
    # Bottom-quartile cutoff.
    return values[len(values) // 4]


def _run_on_clean_sentences(pipeline, sentences, mode: str) -> list[dict]:
    """Run the corrector on clean (unmodified) gold sentences.

    Every token here is correct by construction — there is no injected error.
    Returns the per-token decision trace. `mode` is "binary" or "three_way".
    """
    trace: list[dict] = []
    for sent in sentences:
        text = " ".join(t.form for t in sent.tokens)
        results = pipeline.process(text)
        # Zip pipeline output to gold tokens positionally. They share the same
        # tokeniser by construction (we just joined the gold forms with spaces),
        # so alignment is exact.
        for tok, res in zip(sent.tokens, results):
            if mode == "binary":
                d = correct_token_binary(res)
            else:
                d = correct_token_3way(res)
            trace.append({
                "form": tok.form,
                "gold": tok.form,          # clean text → gold == input
                "upos": tok.upos,
                "decision": d.decision,
                "predicted": d.predicted,
                "confidence": d.confidence,
                "was_corrupted": False,    # clean text: nothing was corrupted
            })
    return trace


def _fcr_from_trace(trace: list[dict], freq: dict[str, int], rare_thr: int) -> dict:
    """Compute overall FCR + stratified breakdown from a decision trace.

    Reports two parallel cuts of "where do false corrections land":
      by_class    — lexicon-miss taxonomy (not_in_lexicon / common / proper_name
                    / borrowing), the surface-form-against-lemma-list view.
      by_analyzer — analyzer-based taxonomy (rma_known / rma_unknown), the
                    structurally meaningful "could the analyser parse it" view.
    Reporting both exposes that the large not_in_lexicon bucket (≈71% of
    tokens) is a property of the narrow frequency lexicon, not of language
    coverage (rma_unknown is ≈0.2%).
    """
    n_total = len(trace)
    decisions = Counter(t["decision"] for t in trace)
    n_changed = decisions.get("CORRECT", 0)
    fcr = n_changed / n_total if n_total else 0.0

    # Stratified FCR: for each class, count changed / total within that class.
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "changed": 0})
    by_analyzer: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "changed": 0})
    by_rarity = {"rare": {"total": 0, "changed": 0}, "common": {"total": 0, "changed": 0}}
    for t in trace:
        cls = _token_class(t["form"], t["upos"], freq)
        by_class[cls]["total"] += 1
        if t["decision"] == "CORRECT":
            by_class[cls]["changed"] += 1
        acl = _analyzer_class(t["form"])
        by_analyzer[acl]["total"] += 1
        if t["decision"] == "CORRECT":
            by_analyzer[acl]["changed"] += 1
        rare = _is_rare(t["form"], freq, rare_thr)
        key = "rare" if rare else "common"
        by_rarity[key]["total"] += 1
        if t["decision"] == "CORRECT":
            by_rarity[key]["changed"] += 1

    def _fmt(d):
        return {
            cls: {
                "total": v["total"],
                "changed": v["changed"],
                "fcr": v["changed"] / v["total"] if v["total"] else 0.0,
            }
            for cls, v in d.items()
        }

    return {
        "n_tokens": n_total,
        "n_changed": n_changed,
        "fcr": fcr,
        "decision_mix": dict(decisions),
        "by_class": _fmt(by_class),
        "by_analyzer": _fmt(by_analyzer),
        "by_rarity": _fmt(by_rarity),
    }


def evaluate_fcr() -> dict:
    """Run FCR evaluation for both binary and three-way configurations.

    Evaluated on folds 1..N-1 only — fold 0 is held out as the calibration
    fold for the decision thresholds (τ_e, τ_c, τ_m), exactly as in
    eval_correction. Including fold 0 in the FCR evaluation would test on the
    fold the thresholds were tuned on; for the binary corrector (which is
    threshold-sensitive) this matters, for three_way it is inert (FCR is 0 on
    every fold). The per-fold vector therefore has N-1 entries, not N.
    """
    pipeline = build_pipeline()
    folds = make_folds()
    freq = get_frequency_dict()
    rare_thr = _rare_threshold(freq)

    # Evaluate on folds 1..N-1 (skip calibration fold 0). The per-fold vector
    # thus has N-1 entries and matches the correction-evaluation protocol.
    eval_fold_indices = list(range(1, len(folds)))

    binary_fcrs: list[float] = []
    threeway_fcrs: list[float] = []

    for fold_idx in eval_fold_indices:
        _train, dev = folds[fold_idx]
        bin_trace = _run_on_clean_sentences(pipeline, dev, "binary")
        thw_trace = _run_on_clean_sentences(pipeline, dev, "three_way")
        bin_metrics = _fcr_from_trace(bin_trace, freq, rare_thr)
        thw_metrics = _fcr_from_trace(thw_trace, freq, rare_thr)
        binary_fcrs.append(bin_metrics["fcr"])
        threeway_fcrs.append(thw_metrics["fcr"])

    # For the stratified breakdown we pool only the evaluation folds' dev
    # halves (folds 1..N-1), which together cover the non-calibration pool.
    all_dev = [s for i in eval_fold_indices for s in folds[i][1]]
    bin_pooled = _run_on_clean_sentences(pipeline, all_dev, "binary")
    thw_pooled = _run_on_clean_sentences(pipeline, all_dev, "three_way")

    return {
        "calibration_fold": 0,
        "eval_folds": eval_fold_indices,
        "n_folds": len(eval_fold_indices),
        "n_test_sentences": len(all_dev),
        "rare_frequency_threshold": rare_thr,
        "binary": {
            "fcr_per_fold": binary_fcrs,
            **_fcr_from_trace(bin_pooled, freq, rare_thr),
        },
        "three_way": {
            "fcr_per_fold": threeway_fcrs,
            **_fcr_from_trace(thw_pooled, freq, rare_thr),
        },
    }


def main() -> None:
    config.ensure_dirs()
    result = evaluate_fcr()
    out_path = os.path.join(config.RAW_DIR, "fcr.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print("N1 — False-correction rate on unmodified normative text")
    print("=" * 64)
    print(f"Folds: calibration fold 0 held out, evaluated on {result['n_folds']} folds "
          f"({result['eval_folds']})  |  test sentences: {result['n_test_sentences']}")
    print(f"Rare-frequency threshold (bottom quartile): {result['rare_frequency_threshold']}")
    print()
    for mode in ("binary", "three_way"):
        m = result[mode]
        fcrs = m["fcr_per_fold"]
        mean = sum(fcrs) / len(fcrs) if fcrs else 0.0
        print(f"  [{mode}]")
        print(f"    FCR per fold: {[round(f, 4) for f in fcrs]}")
        print(f"    FCR mean:     {mean:.2%}   ({m['n_changed']}/{m['n_tokens']} pooled)")
        print(f"    Decision mix: {m['decision_mix']}")
        print(f"    By class (lexicon-miss cut):")
        for cls, v in sorted(m["by_class"].items(), key=lambda x: -x[1]['total']):
            print(f"      {cls:16} FCR={v['fcr']:.2%}  ({v['changed']}/{v['total']})")
        print(f"    By analyzer (parse-success cut):")
        for cls, v in sorted(m["by_analyzer"].items(), key=lambda x: -x[1]['total']):
            print(f"      {cls:16} FCR={v['fcr']:.2%}  ({v['changed']}/{v['total']})")
        print(f"    By rarity:")
        for k, v in m["by_rarity"].items():
            print(f"      {k:14} FCR={v['fcr']:.2%}  ({v['changed']}/{v['total']})")
        print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
