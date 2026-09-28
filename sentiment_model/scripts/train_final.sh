#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
DATA=${DATA:?Set DATA to attachment 2 aligned_50.pkl}
BERT_DIR=${BERT_DIR:?Set BERT_DIR to the local bert-base-uncased directory}
OUT=${OUT:-$ROOT/outputs/retrain}

test -f "$DATA"
test -f "$BERT_DIR/config.json"
test -f "$BERT_DIR/model.safetensors"
test -x "$PYTHON"
if [[ -e "$OUT" ]]; then
    echo "Output already exists: $OUT" >&2
    exit 1
fi
mkdir -p "$OUT"

"$PYTHON" -m q2.run audit --data "$DATA" --output "$OUT/audit.json"
for seed in 3407 42 2026; do
    "$PYTHON" -m q2.run train \
        --data "$DATA" --output "$OUT/backbone" --variant M4 \
        --seed "$seed" --epochs 30 --patience 5 --batch-size 32 \
        --lr 3e-4 --text-encoder bert_base --bert "$BERT_DIR" --class-weight
    "$PYTHON" -m q2.conditional_reg \
        --data "$DATA" --base-checkpoint "$OUT/backbone/M4/$seed/best.pt" \
        --bert "$BERT_DIR" --cache "$OUT/cache_$seed.pt" \
        --output "$OUT/conditional/seed_$seed" --seed "$seed" \
        --head-type binary --strong-threshold 1.5 \
        --gate-loss-weight 0.1 --expert-loss-weight 0.2 \
        --epochs 30 --patience 6 --batch-size 96 --lr 3e-4
    "$PYTHON" -m q2.run grid \
        --data "$DATA" --checkpoint "$OUT/conditional/seed_$seed/best.pt" \
        --bert "$BERT_DIR" --output "$OUT/conditional/seed_$seed/grid"
done

"$PYTHON" -m q2.ensemble --inputs \
    "$OUT/conditional/seed_3407/grid/predictions.csv" \
    "$OUT/conditional/seed_42/grid/predictions.csv" \
    "$OUT/conditional/seed_2026/grid/predictions.csv" \
    --output "$OUT/ensemble"
