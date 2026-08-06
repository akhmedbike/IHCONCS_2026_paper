"""Run the full IHCONCS 2026 evaluation pipeline end-to-end.

Executes every experiment in dependency order, then assembles the tables.
Each step is idempotent: re-running resumes from the JSON on disk where it
makes sense, and overwrites otherwise. Total wall time on the reference
machine (M-series Mac, CPU inference of KazBERT) is ~15–25 minutes, dominated
by N1 and N4 (neural POS inference over the full gold pool × folds).

Usage:
    python experiments/run_all.py            # run everything
    python experiments/run_all.py --skip N4  # skip a named step
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# (step_id, description, module path) in execution order. The order is fixed
# by data dependencies: N2 and N1 need nothing; N3 needs corrupt; N4 needs
# corrupt + the corrector; N5 needs the candidate generator; tables need all.
STEPS = [
    ("N2",  "Analyser coverage",            "experiments.eval_coverage"),
    ("N1",  "False-correction rate (clean)", "experiments.eval_fcr"),
    ("N3",  "Candidate Recall@K",            "experiments.eval_candidate_recall"),
    ("N4",  "Correction P/R/F0.5/CER/WER",   "experiments.eval_correction"),
    ("N5",  "Risk–coverage + Fig.5",         "experiments.make_risk_coverage"),
    ("T",   "Assemble tables",               "scripts.make_tables"),
]


def main() -> None:
    skip = set()
    if "--skip" in sys.argv:
        idx = sys.argv.index("--skip")
        skip = set(sys.argv[idx + 1:])
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only = set(sys.argv[idx + 1:])
        steps = [s for s in STEPS if s[0] in only]
    else:
        steps = STEPS

    for step_id, desc, module in steps:
        if step_id in skip:
            print(f"[{step_id}] SKIP — {desc}")
            continue
        print(f"\n{'=' * 64}\n[{step_id}] {desc}\n{'=' * 64}")
        t0 = time.time()
        # Run as a subprocess so each step gets a fresh interpreter (avoids
        # state carryover between neural model loads and the matplotlib import).
        py = os.path.join(_ROOT, "bin", "py")
        if not os.path.exists(py):
            py = sys.executable
        cmd = [py, "-m", module]
        result = subprocess.run(cmd, cwd=_ROOT)
        dt = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
        print(f"[{step_id}] {status} in {dt:.0f}s")
        if result.returncode != 0:
            print("Aborting pipeline. Fix the failing step and re-run.")
            sys.exit(result.returncode)
    print("\nAll steps complete. See results/tables/all_tables.md.")


if __name__ == "__main__":
    main()
