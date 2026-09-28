#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
DATA=${DATA:?Set DATA to attachment 2 aligned_50.pkl}
BERT_DIR=${BERT_DIR:?Set BERT_DIR to the local bert-base-uncased directory}
SPLIT=${SPLIT:-valid}
WEIGHTS_DIR=${WEIGHTS_DIR:-$ROOT/weights}
OUT=${OUT:-$ROOT/outputs/eval_$SPLIT}

case "$SPLIT" in valid|test) ;; *) echo "SPLIT must be valid or test" >&2; exit 1 ;; esac
test -f "$DATA"
test -f "$BERT_DIR/model.safetensors"
test -x "$PYTHON"
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
    "$PYTHON" -m q2.run grid \
        --data "$DATA" --split "$SPLIT" \
        --checkpoint "$checkpoint" --bert "$BERT_DIR" \
        --output "$OUT/seed_$seed"
done

"$PYTHON" -m q2.ensemble --inputs \
    "$OUT/seed_3407/predictions.csv" \
    "$OUT/seed_42/predictions.csv" \
    "$OUT/seed_2026/predictions.csv" \
    --output "$OUT/ensemble"

"$PYTHON" "$ROOT/scripts/summarize_grid.py" "$OUT/ensemble/metrics.csv"
