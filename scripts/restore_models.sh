#!/usr/bin/env bash
# Restore the trained model weights into models/ (they are not committed).
#
# The KazBERT POS and lemma checkpoints were fine-tuned in the companion
# kazmorph-2026 project. This script tries, in order:
#   1. A sibling kazmorph-2026 checkout (the default co-location).
#   2. The path given by $KAZMORPH_2026.
#   3. Bails out with instructions for manual download.
#
# Only the weight files are missing from a fresh clone — config.json,
# tokenizer.json, and calibration metadata ARE committed, so after this
# script the model directories are complete.
#
# Usage:
#   scripts/restore_models.sh
#   KAZMORPH_2026=/path/to/kazmorph-2026 scripts/restore_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAZ="${KAZMORPH_2026:-$ROOT/../kazmorph-2026}"

# Find the best available POS checkpoint in the kazmorph-2026 tree. The
# *_nosilver_backup is the documented source (5-fold CV F1 ≈ 0.885); fall
# back to the live models/neural_pos if present.
find_pos_checkpoint() {
    local k="$1"
    for c in \
        "$k/models/neural_pos.kazbert_nosilver_backup" \
        "$k/models/neural_pos" \
        "$k/models/backup_old_split/neural_pos"; do
        if [ -f "$c/model.safetensors" ]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

if [ ! -d "$KAZ" ]; then
    cat >&2 <<EOF
restore_models.sh: kazmorph-2026 not found at $KAZ

The model weights are fine-tuned there and not redistributed with this repo.
Options:
  1. Clone kazmorph-2026 as a sibling:  git clone <kazmorph-2026-url> ../kazmorph-2026
  2. Point at an existing copy:         KAZMORPH_2026=/path/to/kazmorph-2026 $0
  3. Re-fine-tune from Eraly-ml/KazBERT (see kazmorph-2026/scripts/train_neural.py).
EOF
    exit 1
fi

POS_SRC="$(find_pos_checkpoint "$KAZ")" || {
    echo "restore_models.sh: no neural_pos checkpoint with model.safetensors under $KAZ/models" >&2
    exit 1
}
LEMMA_SRC="$KAZ/models/neural_lemma"
CRF_SRC="$KAZ/models/crf_model.joblib"

echo "POS checkpoint:   $POS_SRC"
echo "Lemma checkpoint: $LEMMA_SRC"

mkdir -p "$ROOT/models/neural_pos" "$ROOT/models/neural_lemma"
cp -f "$POS_SRC/model.safetensors" "$ROOT/models/neural_pos/"
cp -f "$LEMMA_SRC/model.safetensors" "$ROOT/models/neural_lemma/"
[ -f "$CRF_SRC" ] && cp -f "$CRF_SRC" "$ROOT/models/" && echo "CRF: copied"

echo
echo "Restored. Verify with:"
echo "  ./bin/py -c \"import sys; sys.path.insert(0,'.'); from experiments._pipeline import build_pipeline; p=build_pipeline(); print('neural:', p.neural.model is not None, '| lemma:', p.lemmatizer.model is not None)\""
