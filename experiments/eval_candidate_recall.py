"""N3 — Candidate-generation Recall@K (paper Table 3, eq. 20).

For every corrupted token, we ask: did the generator place the gold (normative)
surface form within its top-K candidates? Recall@K is the share of corrupted
tokens for which the answer is yes. This is a property of the *generator*
alone — no ranker, no thresholds, no pipeline — so it is the cleanest
comparison of the three candidate modes (dl / dl_freq / dl_fst_filter) and the
single most direct test of the paper's claim that morphological filtering
improves candidate quality.

The evaluation is seed-stratified: we corrupt each fold's dev half with
N_CORRUPTION_SEEDS independent seeds and average Recall@K over (fold × seed).
This is the only experiment where multiple seeds matter, because the
corruption operator is stochastic and Recall@K is sensitive to which tokens
happen to be corrupted.

A subtlety worth noting: for character-level corruptions (deletion etc.) the
gold form is always within edit distance 1–2 of the input by construction, so
the SymSpell index (max_dist=2) is in principle able to retrieve it. Recall@K
therefore measures *coverage of the ranking*, not of the retrieval radius —
the interesting failures are tokens where the gold form exists in the index
but is ranked outside top-K by ties or by the frequency/morphology ordering.

Run:  python experiments/eval_candidate_recall.py
Writes: results/raw/candidate_recall.json
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from experiments._pipeline import make_folds  # noqa: E402
from experiments.corrupt import corrupt_fold  # noqa: E402
from experiments.candidate_generator import generate_candidates  # noqa: E402
from experiments.correction_metrics import recall_at_k, rank_of, mean_reciprocal_rank  # noqa: E402


_MODES = ("dl", "dl_freq", "dl_fst_filter", "dl_freq_morph")


def _evaluate_one_pass(pairs, mode: str) -> dict:
    """Compute Recall@K and MRR over one set of corrupted pairs, one mode."""
    # Per-token hit indicators, one list per K. We resolve rank once and derive
    # every K from it, so we scan candidates only once per token.
    ranks: list[int] = []
    n_evaluated = 0
    n_gold_in_index = 0  # tokens where the gold form was retrieved at all
    for p in pairs:
        if p.is_identity or p.corruption is None:
            continue
        c = p.corruption
        # The corrupted form is the generator's input; the gold is the original.
        cands = generate_candidates(c.corrupted, mode=mode, max_candidates=10)
        cand_words = [w for w, _ in cands]
        r = rank_of(c.original, cand_words)
        ranks.append(r)
        n_evaluated += 1
        if r > 0:
            n_gold_in_index += 1

    recall = {}
    for k in config.RECALL_K_VALUES:
        hits = [1 if 0 < r <= k else 0 for r in ranks]
        recall[f"recall_at_{k}"] = recall_at_k([bool(h) for h in hits], k)
    return {
        **recall,
        "mrr": mean_reciprocal_rank(ranks),
        "n_evaluated": n_evaluated,
        "n_gold_in_index": n_gold_in_index,
        "index_coverage": n_gold_in_index / n_evaluated if n_evaluated else 0.0,
    }


def evaluate_candidate_recall() -> dict:
    """Run Recall@K over folds × corruption seeds for every mode."""
    folds = make_folds()
    # results[mode] = { "per_fold_seed": [ {metrics} ], "mean": {...}, "std": {...} }
    results: dict[str, dict] = {}
    for mode in _MODES:
        per_run: list[dict] = []
        for fold_idx, (_train, dev) in enumerate(folds):
            for seed_idx, seed in enumerate(config.CORRUPTION_SEEDS):
                # Combine fold index into the seed so each (fold, seed) pair
                # gets a distinct, reproducible RNG state.
                rng_seed = seed * 1000 + fold_idx
                pairs = corrupt_fold(dev, seed=rng_seed)
                metrics = _evaluate_one_pass(pairs, mode)
                metrics["fold"] = fold_idx
                metrics["seed"] = seed
                per_run.append(metrics)
        # Aggregate scalar metrics across all runs.
        scalar_keys = [f"recall_at_{k}" for k in config.RECALL_K_VALUES] + ["mrr", "index_coverage"]
        agg = {}
        for key in scalar_keys:
            vals = [r[key] for r in per_run]
            mean = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
            agg[key] = {"mean": mean, "std": var ** 0.5, "n": len(vals)}
        results[mode] = {"per_run": per_run, "aggregate": agg}

    # Per-category breakdown for the proposed mode (dl_freq_morph) and, for
    # backward comparability, the dl_fst_filter mode. This shows the paper
    # exactly which corruption types the candidate generator handles and
    # where it fails — the affix-category operators are the hard floor.
    per_category: dict[str, dict[str, dict]] = {}
    for br_mode in ("dl_freq_morph", "dl_fst_filter"):
        cat_breakdown: dict[str, dict] = {}
        for fold_idx, (_train, dev) in enumerate(folds):
            rng_seed = config.CORRUPTION_SEEDS[0] * 1000 + fold_idx
            pairs = corrupt_fold(dev, seed=rng_seed)
            for p in pairs:
                if p.is_identity or p.corruption is None:
                    continue
                cat = p.corruption.category
                bucket = cat_breakdown.setdefault(cat, {"ranks": [], "n": 0})
                cands = generate_candidates(p.corruption.corrupted, mode=br_mode, max_candidates=10)
                bucket["ranks"].append(rank_of(p.corruption.original, [w for w, _ in cands]))
                bucket["n"] += 1
        cat_summary = {}
        for cat, b in cat_breakdown.items():
            cat_summary[cat] = {
                "n": b["n"],
                "recall_at_1": recall_at_k([0 < r <= 1 for r in b["ranks"]], 1),
                "recall_at_5": recall_at_k([0 < r <= 5 for r in b["ranks"]], 5),
                "recall_at_10": recall_at_k([0 < r <= 10 for r in b["ranks"]], 10),
                "mrr": mean_reciprocal_rank(b["ranks"]),
            }
        per_category[br_mode] = cat_summary

    return {
        "n_folds": len(folds),
        "n_seeds": len(config.CORRUPTION_SEEDS),
        "modes": {m: results[m]["aggregate"] for m in _MODES},
        "per_category_dl_fst_filter": per_category["dl_fst_filter"],
        "per_category_dl_freq_morph": per_category["dl_freq_morph"],
    }


def main() -> None:
    config.ensure_dirs()
    result = evaluate_candidate_recall()
    out_path = os.path.join(config.RAW_DIR, "candidate_recall.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print("N3 — Candidate-generation Recall@K")
    print(f"folds={result['n_folds']} × seeds={result['n_seeds']}")
    print("=" * 64)
    print(f"{'Mode':18} {'Rec@1':>8} {'Rec@5':>8} {'Rec@10':>8} {'MRR':>8} {'IdxCov':>8}")
    print("-" * 64)
    for mode in _MODES:
        a = result["modes"][mode]
        print(f"{mode:18} "
              f"{a['recall_at_1']['mean']:7.1%}±{a['recall_at_1']['std']:.1%} "
              f"{a['recall_at_5']['mean']:7.1%}±{a['recall_at_5']['std']:.1%} "
              f"{a['recall_at_10']['mean']:7.1%}±{a['recall_at_10']['std']:.1%} "
              f"{a['mrr']['mean']:7.1%}±{a['mrr']['std']:.1%} "
              f"{a['index_coverage']['mean']:7.1%}")
    print()
    for br_mode, label in (("dl_freq_morph", "PROPOSED (dl_freq_morph)"),
                           ("dl_fst_filter", "dl_fst_filter")):
        key = f"per_category_{br_mode}"
        print(f"Per-category breakdown ({label}, seed 0):")
        print(f"{'Category':32} {'N':>5} {'Rec@1':>7} {'Rec@5':>7} {'Rec@10':>7} {'MRR':>7}")
        print("-" * 70)
        for cat, b in sorted(result[key].items(), key=lambda x: -x[1]["n"]):
            print(f"{cat:32} {b['n']:5} {b['recall_at_1']:6.1%} {b['recall_at_5']:6.1%} "
                  f"{b['recall_at_10']:6.1%} {b['mrr']:6.1%}")
        print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
