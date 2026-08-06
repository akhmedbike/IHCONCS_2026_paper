"""N5 — Risk–coverage curve for the three-way decision rule (paper Fig. 5).

The risk–coverage curve is the single most persuasive artefact for the REVIEW
mechanism: it shows directly that, by routing low-margin tokens to REVIEW
instead of forcing an automatic decision, the system can hold risk (incorrect
automatic decisions / automatic decisions) arbitrarily low at the cost of
coverage (automatic decisions / all tokens). A purely binary corrector has no
such knob — its operating point is fixed.

We compute the curve by sweeping a single confidence threshold τ over the
top-1 candidate probability. At each τ, a token is auto-decided iff its top-1
candidate probability ≥ τ; otherwise it is REVIEWed. "Correct" means the
auto-decision's predicted form equals the gold form.

To make the curve independent of N4's calibrated operating point, we re-run
the candidate generator once over the test folds and sweep τ on the stored
per-token (top_prob, is_top1_correct) pairs. The operating point selected in
N4 is marked on the curve.

Run:  python experiments/make_risk_coverage.py
Writes: results/raw/risk_coverage.json  and  results/figures/fig5_risk_coverage.png
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
from experiments.candidate_generator import generate_candidates, candidate_probs  # noqa: E402
from experiments.freq import build_fold_frequency_dict  # noqa: E402
from experiments.correction_metrics import risk_coverage  # noqa: E402


def _collect_top_probs(pipeline, pairs, candidate_mode="dl_freq", freq=None):
    """For every corrupted token, record (top_prob, is_top1_correct).

    `is_top1_correct` is True iff the top-ranked correction candidate equals
    the gold form. We do NOT factor in the KEEP branch here — the risk–coverage
    framing is about the *correction* decision specifically, so we restrict to
    tokens where a correction candidate exists (the population over which an
    automatic CORRECT could be made). Identity-pair tokens and tokens with no
    candidate are excluded from the curve's denominator, with the count
    reported separately so the paper can state the coverage base explicitly.

    `freq` threads a leakage-free per-fold frequency dictionary into the
    candidate ranking (see experiments.freq).
    """
    confidences: list[float] = []
    correct: list[bool] = []
    n_no_candidate = 0
    for pair in pairs:
        if pair.is_identity or pair.corruption is None:
            continue
        c = pair.corruption
        cands = generate_candidates(c.corrupted, mode=candidate_mode, max_candidates=10, freq=freq)
        probs = candidate_probs(cands, include_original=True, original_token=c.corrupted)
        correction_probs = [(w, p) for w, p in probs if w.lower() != c.corrupted.lower()]
        if not correction_probs:
            n_no_candidate += 1
            continue
        top_word, top_prob = correction_probs[0]
        confidences.append(top_prob)
        correct.append(top_word.lower() == c.original.lower())
    return confidences, correct, n_no_candidate


def evaluate_risk_coverage(*, leakage_free: bool = True) -> dict:
    """Sweep the confidence threshold over pooled test-fold predictions."""
    pipeline = build_pipeline()
    folds = make_folds()

    # Pool all test folds (1..N-1) × seeds, exactly as N4 does, so the curve
    # is computed on the same population the headline metrics use.
    all_conf: list[float] = []
    all_correct: list[bool] = []
    n_no_candidate_total = 0
    for fold_idx in range(1, len(folds)):
        dev = folds[fold_idx][1]
        fold_freq = build_fold_frequency_dict(dev) if leakage_free else None
        # Average over seeds by pooling — the curve is a population statistic,
        # so pooling is equivalent to averaging per-seed curves and simpler.
        for seed in config.CORRUPTION_SEEDS:
            pairs = corrupt_fold(dev, seed=seed * 1000 + fold_idx)
            conf, correct, n_nc = _collect_top_probs(pipeline, pairs, freq=fold_freq)
            all_conf.extend(conf)
            all_correct.extend(correct)
            n_no_candidate_total += n_nc

    curve = risk_coverage(all_conf, all_correct, config.TAU_SWEEP)

    # The N4 calibrated operating point (load from correction.json if present).
    operating_point = None
    corr_path = os.path.join(config.RAW_DIR, "correction.json")
    if os.path.exists(corr_path):
        with open(corr_path) as f:
            corr = json.load(f)
        cal = corr.get("calibration", {})
        operating_point = {"tau_c": cal.get("tau_c"), "tau_e": cal.get("tau_e"),
                           "tau_m": cal.get("tau_m")}

    return {
        "n_evaluated": len(all_conf),
        "n_no_candidate": n_no_candidate_total,
        "curve": curve,
        "operating_point": operating_point,
        "thresholds_swept": list(config.TAU_SWEEP),
    }


def plot_curve(result: dict, out_path: str) -> None:
    """Render Fig. 5: risk (y) vs coverage (x), with the operating point marked.

    The operating point is the calibrated τ_c from N4's correction.json; it is
    drawn as a vertical line at the coverage that threshold produces on this
    curve, so the reader can see exactly where the system operates rather than
    reading it only from the caption. PNG (raster, compatibility) plus a
    vector PDF (publication) are written side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["curve"]
    coverages = [p["coverage"] for p in curve]
    risks = [p["risk"] for p in curve]
    thresholds = [p["threshold"] for p in curve]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(coverages, risks, marker="o", linewidth=2, color="#1f5fb4",
            label="three-way (REVIEW routing)")

    # Mark the calibrated operating point (τ_c) as a vertical line at the
    # coverage that threshold yields on this curve. Previously the operating
    # point appeared only in the title; a reviewer reading Fig. 5 in isolation
    # could not tell which point the system actually runs at.
    op = result.get("operating_point")
    if op and op.get("tau_c") is not None:
        tau_c = op["tau_c"]
        # Find the curve point whose threshold is τ_c.
        match = next((p for p in curve if abs(p["threshold"] - tau_c) < 1e-9), None)
        if match is not None:
            ax.axvline(match["coverage"], color="#c0392b", linestyle="--",
                       linewidth=1.2, alpha=0.8)
            ax.plot(match["coverage"], match["risk"], "o", color="#c0392b",
                    markersize=8, zorder=5,
                    label=f"operating point τ={tau_c:g}\n"
                          f"(cov {match['coverage']:.1%}, risk {match['risk']:.1%})")
        title = (f"Risk–coverage (operating point: τ_c={op['tau_c']}, "
                 f"τ_e={op['tau_e']}, τ_m={op['tau_m']})")
        ax.set_title(title, fontsize=9)
    else:
        ax.set_title("Risk–coverage curve", fontsize=11)

    ax.set_xlabel("Coverage C(τ) — share of tokens auto-decided", fontsize=10)
    ax.set_ylabel("Risk R(τ) — error rate among auto-decisions", fontsize=10)
    ax.set_ylim(-0.02, max(0.6, (max(risks) + 0.05) if risks else 1.0))
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    # Raster (compatibility) and vector (publication). The vector PDF keeps
    # text crisp at any zoom and is the format the camera-ready must use.
    fig.savefig(out_path, dpi=200)
    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaky", action="store_true",
                    help="use the global (leaky) frequency dict instead of the per-fold one")
    args = ap.parse_args()
    result = evaluate_risk_coverage(leakage_free=not args.leaky)
    suffix = "_leaky" if args.leaky else ""
    out_json = os.path.join(config.RAW_DIR, f"risk_coverage{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print("N5 — Risk–coverage curve (Fig. 5)")
    print("=" * 64)
    print(f"Tokens evaluated: {result['n_evaluated']} "
          f"(+{result['n_no_candidate']} with no candidate, excluded)")
    print()
    print(f"{'τ':>6} {'coverage':>10} {'risk':>10} {'n_auto':>8} {'n_incorrect':>12}")
    print("-" * 52)
    for p in result["curve"]:
        print(f"{p['threshold']:6.2f} {p['coverage']:10.1%} {p['risk']:10.1%} "
              f"{p['n_auto']:8} {p['n_incorrect']:12}")
    print()
    fig_path = os.path.join(config.FIGURES_DIR, "fig5_risk_coverage.png")
    try:
        plot_curve(result, fig_path)
        print(f"Figure written: {fig_path} (+ .pdf vector)")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"JSON written: {out_json}")


if __name__ == "__main__":
    main()
