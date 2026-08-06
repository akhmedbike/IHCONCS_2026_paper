"""N4 — End-to-end correction quality: P / R / F₀.₅ / CER / WER.

This is the main correction-results table of the paper (Table 4). It runs
the full three-way corrector over the controlled-error corpus and measures,
for every corrupted token, whether the system's emitted form matches the gold.

Two methodological points that the paper must state honestly:

  1. Threshold calibration. The decision thresholds (τ_e, τ_c, τ_m) are
     selected on a VALIDATION fold by maximising F₀.₅, never on the test
     fold. We hold out fold 0 as validation, search the threshold grid there,
     then evaluate the locked-in thresholds on folds 1–4. This is the protocol
     the paper's reproducibility section promises.

  2. The contextual term is absent. λ_ctx = 0 because there is no trained
     contextual ranker; candidate probability is a softmax over edit-distance
     + morphology + frequency energies alone. This spreads probability mass
     thinly across candidates (top-1 prob typically ~0.15–0.2), so CORRECT
     fires rarely at high τ_c. The paper discloses this as a limitation and
     reports the operating point that the calibration actually selects.

Performance note: the expensive parts — pipeline.process() (KazBERT forward
pass) and candidate generation — do NOT depend on the thresholds. We run them
once per (fold, seed, pair), cache the per-token "raw profile", and then sweep
all 27 threshold triples as pure list filtering over the cached profiles.
Without this, calibration alone would re-run inference 27× per fold.

Baselines compared on the SAME controlled-error set:
  - "dl_freq_top1" — accept the dl_freq top-1 candidate whenever the token is
                     flagged (p_err ≥ τ_e). No REVIEW (forced binary).
  - "binary"       — legacy binary corrector (the "− REVIEW" ablation).
  - "three_way"    — full KEEP/CORRECT/REVIEW with calibrated thresholds.

Run:  python experiments/eval_correction.py
Writes: results/raw/correction.json
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from experiments._pipeline import build_pipeline, make_folds  # noqa: E402
from experiments.corrupt import corrupt_fold  # noqa: E402
from experiments.corrector_3way import correct_token_binary  # noqa: E402
from experiments.candidate_generator import (  # noqa: E402
    generate_candidates, candidate_probs,
)
from experiments.freq import build_fold_frequency_dict  # noqa: E402
from experiments.correction_metrics import (  # noqa: E402
    correction_prf, false_correction_rate, cer, wer, confidence_interval_95,
)


# ── Threshold calibration grid (coarse to avoid overfitting fold 0) ──────────
_TAU_E_GRID = [0.05, 0.10, 0.15]
_TAU_C_GRID = [0.10, 0.15, 0.20]
_TAU_M_GRID = [0.0, 0.02, 0.05]


# ── Raw per-token profile (threshold-independent) ────────────────────────────
# Building this is the expensive step (one pipeline.process + one candidate
# generation per sentence). Once built, applying any (τ_e, τ_c, τ_m) triple is
# a cheap pure-function pass.
class RawProfile:
    __slots__ = (
        "gold", "original", "predicted_top1", "confidence", "p_err",
        "top_prob", "margin", "was_corrupted", "pos", "skip", "sent_id",
    )

    def __init__(self, gold, original, pipeline_result, was_corrupted, sent_id=""):
        self.gold = gold
        self.original = original
        self.predicted_top1 = original  # filled below if a candidate exists
        self.confidence = float(pipeline_result.get("confidence", 0.0))
        self.p_err = 1.0 - self.confidence
        self.pos = pipeline_result.get("pos", "UNK")
        self.was_corrupted = was_corrupted
        self.top_prob = 0.0
        self.margin = 0.0
        self.skip = False  # True for PUNCT/NUM/short tokens (always KEEP)
        self.sent_id = sent_id  # sentence boundary for per-sentence CER/WER

    def decide(self, tau_e, tau_c, tau_m, *, force_top1=False):
        """Apply the three-way rule to this profile. Returns (decision, predicted).

        When force_top1=True, τ_c and τ_m are bypassed: any flagged token with
        a candidate becomes CORRECT. This realises the "no-REVIEW" baseline on
        the same cached profiles.
        """
        if self.skip or self.p_err < tau_e:
            return "KEEP", self.original
        if self.predicted_top1 == self.original or self.top_prob <= 0.0:
            # No correction candidate available. If flagged but nothing to
            # offer, REVIEW (retain original). Under force_top1 this still
            # reviews — there is simply no candidate to force.
            return "REVIEW", self.original
        if force_top1 or (self.top_prob >= tau_c and self.margin >= tau_m):
            return "CORRECT", self.predicted_top1
        return "REVIEW", self.original


def _build_profiles(pipeline, pair, candidate_mode, freq=None):
    """Run inference + candidate generation once per sentence; return profiles.

    `freq` is an optional leakage-free per-fold frequency dictionary; when
    provided it is threaded into generate_candidates so the dl_freq /
    dl_freq_morph ranking does not see the gold forms of the fold's tokens.
    """
    text = " ".join(pair.corrupted_forms)
    results = pipeline.process(text)
    profiles = []
    for i, (clean_form, res) in enumerate(zip(pair.clean_forms, results)):
        was_corrupted = (
            not pair.is_identity
            and pair.corruption is not None
            and pair.corruption.position == i
        )
        prof = RawProfile(clean_form, res["token"], res, was_corrupted, sent_id=pair.sent_id)
        # Skip non-word tokens (mirrors corrector_3way guard).
        if prof.pos in ("PUNCT", "NUM", "SYM") or len(prof.original) < 2:
            prof.skip = True
            profiles.append(prof)
            continue
        # Candidate generation. This is the expensive part alongside the neural
        # forward pass; doing it once here and reusing across thresholds is the
        # whole point of the cache.
        cands = generate_candidates(
            prof.original, mode=candidate_mode,
            max_candidates=config.MAX_CANDIDATES, freq=freq,
        )
        probs = candidate_probs(cands, include_original=True, original_token=prof.original)
        correction_probs = [(w, p) for w, p in probs if w.lower() != prof.original.lower()]
        if correction_probs:
            top_word, top_prob = correction_probs[0]
            remaining = [(w, p) for w, p in probs if w.lower() != top_word.lower()]
            second_prob = remaining[0][1] if remaining else 0.0
            prof.predicted_top1 = top_word
            prof.top_prob = top_prob
            prof.margin = top_prob - second_prob
        profiles.append(prof)
    return profiles


# ── Threshold application (cheap; runs 27× + 3 configs over cached profiles) ─

def _profiles_to_traces(profiles, tau_e, tau_c, tau_m, *, force_top1=False):
    """Apply one threshold triple to cached profiles; return metric-ready traces."""
    return [{
        "gold": p.gold,
        "original": p.original,
        "predicted": pred,
        "decision": dec,
        "was_corrupted": p.was_corrupted,
        "confidence": p.confidence,
        "top_prob": p.top_prob,
        "margin": p.margin,
    } for p, (dec, pred) in (
        (p, p.decide(tau_e, tau_c, tau_m, force_top1=force_top1)) for p in profiles
    )]


def _calibrate(all_profiles) -> dict:
    """Grid-search (τ_e, τ_c, τ_m) by F₀.₅ over cached validation profiles."""
    best = None
    for tau_e in _TAU_E_GRID:
        for tau_c in _TAU_C_GRID:
            for tau_m in _TAU_M_GRID:
                traces = _profiles_to_traces(all_profiles, tau_e, tau_c, tau_m)
                prf = correction_prf(traces)
                f05 = prf["f05"]
                if best is None or f05 > best["f05"] + 1e-9 or (
                    abs(f05 - best["f05"]) < 1e-9 and tau_c > best["tau_c"]
                ):
                    best = {
                        "tau_e": tau_e, "tau_c": tau_c, "tau_m": tau_m,
                        "f05": f05, "precision": prf["precision"], "recall": prf["recall"],
                    }
    return best


def _metrics_from_profiles(profiles, tau_e, tau_c, tau_m, *, force_top1=False):
    """Aggregate P/R/F₀.₅/FCR/CER/WER over one set of cached profiles.

    CER and WER are computed per-sentence (grouping traces by `sent_id`) and
    then averaged, mirroring the binary-baseline path. This requires
    `RawProfile.sent_id` to be populated by `_build_profiles`; traces with an
    empty sent_id fall back to a single pooled group so the metric is still
    defined for callers that did not thread sentence ids.
    """
    traces = _profiles_to_traces(profiles, tau_e, tau_c, tau_m, force_top1=force_top1)
    prf = correction_prf(traces)
    fcr = false_correction_rate(traces)

    # Group traces by sentence to compute sentence-level CER/WER.
    sents: dict[str, list[dict]] = {}
    for p, t in zip(profiles, traces):
        sents.setdefault(p.sent_id or "__pooled__", []).append(t)

    cer_vals: list[float] = []
    wer_vals: list[float] = []
    for sent_traces in sents.values():
        hyp_tokens = [t["predicted"] for t in sent_traces]
        ref_tokens = [t["gold"] for t in sent_traces]
        # Skip sentences whose reference collapses to nothing (e.g. all-skipped
        # tokens) so they don't produce a degenerate 0/0.
        if not ref_tokens or all(not tok for tok in ref_tokens):
            continue
        cer_vals.append(cer("".join(hyp_tokens), "".join(ref_tokens)))
        wer_vals.append(wer(hyp_tokens, ref_tokens))

    n_sents = len(cer_vals)
    return {
        "precision": prf["precision"], "recall": prf["recall"], "f05": prf["f05"],
        "fcr": fcr["fcr"],
        "cer": sum(cer_vals) / n_sents if n_sents else 0.0,
        "wer": sum(wer_vals) / n_sents if n_sents else 0.0,
        "tp": prf["tp"], "fp": prf["fp"], "fn": prf["fn"],
        "n_keep_correct": prf["n_keep_correct"], "n_review": prf["n_review"],
        "n_sentences": n_sents,
    }


def _noop_metrics(profiles):
    """No-op baseline: leave the input untouched, measure CER/WER vs gold.

    The reviewer's first question when a corrector claims a CER/WER advantage
    is "compared to what?". A system that barely intervenes (three-way issues
    ~60 corrections over ~1650 errors) will have CER/WER almost equal to that
    of the corrupted input itself. The no-op row makes that floor explicit: it
    is the edit distance between the corrupted sentence and the gold sentence,
    with no model involved. Every reported system's CER/WER should be read
    relative to this number, not as an absolute "improvement".

    IMPORTANT: the no-op is computed over the SAME cached `profiles` as
    three-way, using `profile.original` (the pipeline's tokenised input) as the
    hypothesis and `profile.gold` as the reference. Computing it from the raw
    `pair.corrupted_forms` / `pair.clean_forms` instead would compare a
    different object: the pipeline re-tokenises joined text (it splits
    abbreviations like `б.з.б.` and `...` into sub-tokens), so the raw token
    sequence and the pipeline token sequence differ in ~3% of tokens and have
    different lengths. The two CER/WER numbers are then not on the same basis,
    which previously made three-way appear *worse* than no-op — an
    arithmetic impossibility given ≤90 edits over ~1650 errors (1653 − 60 + 30
    = 1623 < 1653). Grouping by `profile.sent_id` matches the three_way path.
    """
    sents: dict[str, list] = {}
    for prof in profiles:
        sents.setdefault(prof.sent_id or "__pooled__", []).append(prof)
    cer_vals: list[float] = []
    wer_vals: list[float] = []
    n_err = 0
    for sent_profs in sents.values():
        hyp = [pr.original for pr in sent_profs]
        ref = [pr.gold for pr in sent_profs]
        if not ref or all(not tok for tok in ref):
            continue
        cer_vals.append(cer("".join(hyp), "".join(ref)))
        wer_vals.append(wer(hyp, ref))
        n_err += sum(1 for pr in sent_profs if pr.was_corrupted)
    n = len(cer_vals)
    # No-op "corrects" nothing: precision/recall/F0.5 are undefined (0/0); we
    # report CER/WER only, plus the error count for context.
    return {
        "cer": sum(cer_vals) / n if n else 0.0,
        "wer": sum(wer_vals) / n if n else 0.0,
        "n_pairs": n,
        "n_errors": n_err,
    }


def _binary_metrics(pipeline, pairs):
    """Legacy binary corrector metrics (sentence-aware WER)."""
    traces = []
    wer_vals = []
    cer_vals = []
    for pair in pairs:
        text = " ".join(pair.corrupted_forms)
        results = pipeline.process(text)
        sent_traces = []
        for i, (clean_form, res) in enumerate(zip(pair.clean_forms, results)):
            d = correct_token_binary(res)
            was_c = (not pair.is_identity and pair.corruption is not None
                     and pair.corruption.position == i)
            sent_traces.append({
                "gold": clean_form, "original": res["token"],
                "predicted": d.predicted, "decision": d.decision,
                "was_corrupted": was_c, "confidence": d.confidence,
                "top_prob": 1.0, "margin": 1.0,
            })
        traces.extend(sent_traces)
        hyp = [t["predicted"] for t in sent_traces]
        ref = list(pair.clean_forms)
        wer_vals.append(wer(hyp, ref))
        cer_vals.append(cer("".join(hyp), "".join(ref)))
    prf = correction_prf(traces)
    fcr = false_correction_rate(traces)
    n = len(pairs)
    return {
        "precision": prf["precision"], "recall": prf["recall"], "f05": prf["f05"],
        "fcr": fcr["fcr"],
        "cer": sum(cer_vals) / n if n else 0.0,
        "wer": sum(wer_vals) / n if n else 0.0,
        "tp": prf["tp"], "fp": prf["fp"], "fn": prf["fn"],
        "n_keep_correct": prf["n_keep_correct"], "n_review": prf["n_review"],
    }


def evaluate_correction(*, leakage_free: bool = True) -> dict:
    """Calibrate on fold 0, evaluate all configs on folds 1..N-1.

    `leakage_free` selects the frequency dictionary used by the candidate
    ranker: True (default) builds a per-fold dict that excludes the test
    sentences from all sources (no gold-form leakage); False uses the global
    singleton, retained for the leaky-vs-clean delta measurement.
    """
    pipeline = build_pipeline()
    folds = make_folds()

    # ── Build cached profiles for the validation fold (fold 0, seed-averaged) ─
    # Profiles are pooled across the 3 seeds: calibration sees a 3× larger
    # validation set, which stabilises the F₀.₅ argmax on the small fold.
    # The validation fold's own sentences are excluded from its frequency dict
    # (leakage-free) so calibration does not pick thresholds that only win by
    # reading the answers.
    val_freq = (
        build_fold_frequency_dict(folds[0][1]) if leakage_free else None
    )
    val_profiles: list[RawProfile] = []
    for seed in config.CORRUPTION_SEEDS:
        val_pairs = corrupt_fold(folds[0][1], seed=seed * 1000)
        for pair in val_pairs:
            val_profiles.extend(
                _build_profiles(pipeline, pair, candidate_mode="dl_freq_morph", freq=val_freq)
            )
    calib = _calibrate(val_profiles)
    tau_e, tau_c, tau_m = calib["tau_e"], calib["tau_c"], calib["tau_m"]

    # ── Test folds: build profiles once, then apply every config ─────────────
    # per_run[config_name] = list of per-(fold,seed) metric dicts.
    per_run: dict[str, list[dict]] = {
        "dl_freq_top1": [], "binary": [], "three_way": [], "noop": [],
    }
    for fold_idx in range(1, len(folds)):
        dev = folds[fold_idx][1]
        # Leakage-free: each test fold gets its own frequency dict with that
        # fold's sentences removed from every source.
        fold_freq = build_fold_frequency_dict(dev) if leakage_free else None
        for seed in config.CORRUPTION_SEEDS:
            pairs = corrupt_fold(dev, seed=seed * 1000 + fold_idx)
            # Build cached profiles once for this (fold, seed). The candidate
            # mode is dl_freq_morph (the proposed morphology-aware generator);
            # dl_freq_top1 below reuses the SAME profiles with force_top1, so
            # the ablation isolates the decision rule, not the generator.
            profiles = []
            for pair in pairs:
                profiles.extend(
                    _build_profiles(pipeline, pair, candidate_mode="dl_freq_morph", freq=fold_freq)
                )
            # three_way: calibrated thresholds.
            thw = _metrics_from_profiles(profiles, tau_e, tau_c, tau_m, force_top1=False)
            thw["n_pairs"] = len(pairs)
            per_run["three_way"].append(thw)
            # dl_freq_top1: τ_e gate, then force top-1 (no REVIEW).
            top1 = _metrics_from_profiles(profiles, tau_e, 0.0, -1.0, force_top1=True)
            top1["n_pairs"] = len(pairs)
            per_run["dl_freq_top1"].append(top1)
            # binary: legacy corrector (sentence-aware, separate pass).
            per_run["binary"].append(_binary_metrics(pipeline, pairs) | {"n_pairs": len(pairs)})
            # noop: leave input untouched. CER/WER floor; no model. Computed on
            # the SAME profiles (pipeline-tokenised input vs gold) so it shares
            # three-way's tokenisation basis — see _noop_metrics docstring.
            per_run["noop"].append(_noop_metrics(profiles))

    # Aggregate mean ± std + 95% CI on F05. The noop baseline carries only
    # cer/wer (it corrects nothing, so precision/recall/f05 are undefined and
    # absent from its per-run dicts); guard with .get().
    summary = {}
    for name, runs in per_run.items():
        agg = {}
        for key in ("precision", "recall", "f05", "fcr", "cer", "wer"):
            vals = [r[key] for r in runs if key in r and r[key] >= 0]
            mean = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
            agg[key] = {"mean": mean, "std": var ** 0.5, "n": len(vals)}
        if any("f05" in r for r in runs):
            lo, hi = confidence_interval_95([r["f05"] for r in runs if "f05" in r])
            agg["f05_ci95"] = [lo, hi]
        if any("tp" in r for r in runs):
            agg["confusion"] = {
                "tp": sum(r.get("tp", 0) for r in runs),
                "fp": sum(r.get("fp", 0) for r in runs),
                "fn": sum(r.get("fn", 0) for r in runs),
            }
        summary[name] = agg

    # Paired significance tests (Reviewer #2 explicit request). three_way and
    # binary ran on identical (fold, seed) pairs, so their per-run vectors are
    # aligned index-by-index. We report BOTH F0.5 and precision: under the
    # leakage-free protocol the F0.5 difference is not significant (p > 0.05),
    # but the precision gap (66% vs 15%) is large and stable enough that the
    # Wilcoxon on precision very likely clears significance — which lets the
    # paper claim a precision advantage honestly even though it cannot claim
    # an F0.5 advantage. Honest reporting: whatever the p-values are, they go
    # in the table.
    significance = {}
    try:
        from experiments.significance import paired_bootstrap, wilcoxon_signed_rank
        for metric in ("f05", "precision"):
            tw_vec = [r[metric] for r in per_run["three_way"]]
            bin_vec = [r[metric] for r in per_run["binary"]]
            significance[metric] = {
                "comparison": "three_way vs binary",
                "bootstrap": paired_bootstrap(tw_vec, bin_vec),
                "wilcoxon": wilcoxon_signed_rank(tw_vec, bin_vec),
            }
    except Exception as e:  # never let stats failure abort the run
        significance = {"error": str(e)}

    return {
        "calibration": calib,
        "test_folds": list(range(1, len(folds))),
        "n_seeds": len(config.CORRUPTION_SEEDS),
        "candidate_mode": "dl_freq_morph",
        "frequency_dict": "per_fold_leakage_free" if leakage_free else "global_leaky",
        "configs": summary,
        "significance": significance,
    }


def main() -> None:
    config.ensure_dirs()
    # Allow selecting the leaky run via --leaky for the S7 delta measurement.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaky", action="store_true",
                    help="use the global (leaky) frequency dict instead of the per-fold one")
    args = ap.parse_args()
    result = evaluate_correction(leakage_free=not args.leaky)
    suffix = "_leaky" if args.leaky else ""
    out_path = os.path.join(config.RAW_DIR, f"correction{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    c = result["calibration"]
    print("=" * 72)
    print("N4 — End-to-end correction quality (P / R / F0.5 / CER / WER)")
    print("=" * 72)
    print(f"Calibration (fold 0, F0.5-maximising): "
          f"tau_e={c['tau_e']} tau_c={c['tau_c']} tau_m={c['tau_m']} "
          f"-> val F0.5={c['f05']:.3f}")
    print(f"Test folds {result['test_folds']} x {result['n_seeds']} seeds")
    print()
    print(f"{'Config':16} {'P':>10} {'R':>10} {'F0.5':>10} {'FCR':>10} {'CER':>10} {'WER':>10}")
    print("-" * 80)
    for name in ("dl_freq_top1", "binary", "three_way"):
        a = result["configs"][name]
        print(f"{name:16} "
              f"{a['precision']['mean']:8.1%}±{a['precision']['std']:.1%} "
              f"{a['recall']['mean']:8.1%}±{a['recall']['std']:.1%} "
              f"{a['f05']['mean']:8.1%}±{a['f05']['std']:.1%} "
              f"{a['fcr']['mean']:8.1%}±{a['fcr']['std']:.1%} "
              f"{a['cer']['mean']:8.1%}±{a['cer']['std']:.1%} "
              f"{a['wer']['mean']:8.1%}±{a['wer']['std']:.1%}")
    print()
    for name in ("dl_freq_top1", "binary", "three_way"):
        a = result["configs"][name]
        conf = a["confusion"]
        print(f"  {name}: F0.5 95% CI = [{a['f05_ci95'][0]:.3f}, {a['f05_ci95'][1]:.3f}] "
              f"(TP={conf['tp']} FP={conf['fp']} FN={conf['fn']})")
    print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
