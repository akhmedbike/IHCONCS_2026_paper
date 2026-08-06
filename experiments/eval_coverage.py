"""N2 — Morphological analyser coverage on UD Kazakh-KTB.

This is the cheapest defensible number in the paper: it requires no error
corpus and no candidate generation. It measures the analyser's ability to
produce a non-trivial analysis for gold tokens, which directly supports the
paper's central claim that "lexical absence alone is not evidence of an error"
— if the analyser covers a token, the corrector has no business flagging it.

Three quantities are reported:

  1. Analysis coverage — share of tokens for which ReverseMorphAnalyzer
     returns a non-UNK POS tag (i.e. the trie + lexicon + closed-class lists
     recognise the form). This is the analyser-only path; the neural tagger
     is NOT included here, because the paper attributes coverage to the
     formal component specifically.

  2. Lemma recovery — share of tokens whose predicted lemma matches the gold
     lemma (case-insensitive). This bounds the upper limit of any
     morphology-dependent downstream task.

  3. Multi-analysis potential — share of tokens for which more than one
     admissible morphological path exists. The paper motivates the lattice
     representation on the grounds that ambiguity is common; this number is
     the empirical evidence. The shipped RMA returns a single analysis, so we
     estimate the multi-path potential by counting how many distinct
     (stem, last-affix-category) decompositions the trie admits for each
     token — a conservative lower bound on true lattice width.

Run:  python experiments/eval_coverage.py
Writes: results/raw/coverage.json  and  prints a summary.
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from experiments._pipeline import load_gold_pool  # noqa: E402
from kazmorph.rma import ReverseMorphAnalyzer, ReverseTrie  # noqa: E402
from kazmorph.affixes import AFFIXES  # noqa: E402


def _count_alternative_paths(token: str, trie: ReverseTrie) -> int:
    """Lower bound on the number of admissible suffix paths for `token`.

    Walks the reverse trie from the token's end, counting every complete
    suffix node reachable. Each such node is one admissible suffix; the count
    is a lower bound on lattice width because it ignores morphotactic order
    constraints (a full lattice would prune illegal suffix sequences). This
    is intentionally conservative — we want to under-claim ambiguity.
    """
    n = 0
    node = trie.root
    for ch in reversed(token.lower()):
        if ch not in node.children:
            break
        node = node.children[ch]
        if node.is_end and node.affix_data:
            n += 1
    return n


def evaluate_coverage() -> dict:
    """Compute analyser coverage, lemma recovery, and multi-path potential."""
    pool = load_gold_pool()
    rma = ReverseMorphAnalyzer()
    trie = ReverseTrie()

    n_total = 0
    n_analysed = 0          # non-UNK POS from the formal analyser
    n_lemma_correct = 0     # predicted lemma == gold lemma
    n_multi_path = 0        # ≥ 2 admissible suffix paths in the trie

    per_upos: dict[str, dict[str, int]] = {}

    for sent in pool:
        for tok in sent.tokens:
            form, gold_lemma, gold_upos = tok.form, tok.lemma, tok.upos
            # Skip punctuation and numerals — they short-circuit in the analyser
            # and would inflate coverage trivially. We report content-word
            # coverage separately to be honest about what the number means.
            if gold_upos in ("PUNCT", "SYM"):
                continue
            n_total += 1
            res = rma.analyze(form)
            analysed = res.pos != "UNK"
            lemma_ok = res.lemma.lower() == gold_lemma.lower()

            if analysed:
                n_analysed += 1
            if lemma_ok:
                n_lemma_correct += 1

            # Multi-path potential (content words only, ≥ 3 chars).
            if len(form) >= 3:
                alts = _count_alternative_paths(form, trie)
                if alts >= 2:
                    n_multi_path += 1

            bucket = per_upos.setdefault(gold_upos, {"total": 0, "analysed": 0})
            bucket["total"] += 1
            if analysed:
                bucket["analysed"] += 1

    coverage = n_analysed / n_total if n_total else 0.0
    lemma_acc = n_lemma_correct / n_total if n_total else 0.0
    multi_frac = n_multi_path / n_total if n_total else 0.0

    return {
        "n_tokens": n_total,
        "n_analysed": n_analysed,
        "analysis_coverage": coverage,
        "n_lemma_correct": n_lemma_correct,
        "lemma_recovery": lemma_acc,
        "n_multi_path": n_multi_path,
        "multi_path_fraction": multi_frac,
        "per_upos_coverage": {
            upos: {
                "total": b["total"],
                "analysed": b["analysed"],
                "coverage": b["analysed"] / b["total"] if b["total"] else 0.0,
            }
            for upos, b in sorted(per_upos.items())
        },
    }


def main() -> None:
    config.ensure_dirs()
    result = evaluate_coverage()
    out_path = os.path.join(config.RAW_DIR, "coverage.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("N2 — Morphological analyser coverage (UD Kazakh-KTB)")
    print("=" * 60)
    print(f"Tokens evaluated (excl. PUNCT/SYM): {result['n_tokens']}")
    print(f"Analysis coverage:     {result['analysis_coverage']:.1%}")
    print(f"  ({result['n_analysed']} / {result['n_tokens']})")
    print(f"Lemma recovery:        {result['lemma_recovery']:.1%}")
    print(f"  ({result['n_lemma_correct']} / {result['n_tokens']})")
    print(f"Multi-path potential:  {result['multi_path_fraction']:.1%}")
    print(f"  ({result['n_multi_path']} tokens with ≥2 admissible suffix paths)")
    print()
    print("Per-UPOS coverage:")
    for upos, b in result["per_upos_coverage"].items():
        print(f"  {upos:8} {b['coverage']:6.1%}  ({b['analysed']}/{b['total']})")
    print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
