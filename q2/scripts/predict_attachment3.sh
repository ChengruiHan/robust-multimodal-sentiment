#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
ATTACHMENT3_DIR=${ATTACHMENT3_DIR:?Set ATTACHMENT3_DIR to attachment 3 aligned sample directory}
BERT_DIR=${BERT_DIR:?Set BERT_DIR to the local bert-base-uncased directory}
WEIGHTS_DIR=${WEIGHTS_DIR:-$ROOT/weights}
OUT=${OUT:-$ROOT/outputs/attachment3}

test -x "$PYTHON"
test -f "$BERT_DIR/model.safetensors"
test -d "$ATTACHMENT3_DIR"
"$PYTHON" "$ROOT/scripts/check_assets.py" --root "$ROOT" --bert-dir "$BERT_DIR"
if [[ -e "$OUT" ]]; then
    echo "Output already exists: $OUT" >&2
    exit 1
fi
mkdir -p "$OUT"

for seed in 3407 42 2026; do
    checkpoint="$WEIGHTS_DIR/seed_$seed/best.pt"
    test -f "$checkpoint"
    test -f "$WEIGHTS_DIR/seed_$seed/scaler.npz"
    "$PYTHON" -m q2.run attachment3 \
        --checkpoint "$checkpoint" --bert "$BERT_DIR" \
        --input "$ATTACHMENT3_DIR" --output "$OUT/seed_$seed"
done

"$PYTHON" -m q2.finalize_ensemble --mode attachment3 \
    --inputs \
        "$OUT/seed_3407/attachment3_predictions.csv" \
        "$OUT/seed_42/attachment3_predictions.csv" \
        "$OUT/seed_2026/attachment3_predictions.csv" \
    --diagnostics "$OUT/seed_3407/attachment3_missing_diagnostics.csv" \
    --output "$OUT/ensemble"

"$PYTHON" "$ROOT/scripts/validate_attachment3.py" \
    "$OUT/ensemble/attachment3_predictions.csv"
