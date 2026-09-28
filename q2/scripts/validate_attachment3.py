"""Validate the complete, label-free Q2 submission CSV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FIELDS = ("sample_no", "pred_label", "pred_intensity", "prob_negative",
          "prob_neutral", "prob_positive")
LABELS = {"Negative", "Neutral", "Positive"}


def validate(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"unexpected columns: {reader.fieldnames}")
        rows = list(reader)
    numbers = [int(row["sample_no"]) for row in rows]
    if numbers != list(range(1, 31)):
        raise ValueError("expected exactly samples 1..30 in order")
    for row in rows:
        if row["pred_label"] not in LABELS:
            raise ValueError(f"invalid label at sample {row['sample_no']}")
        intensity = float(row["pred_intensity"])
        probability = [float(row[key]) for key in FIELDS[3:]]
        if not math.isfinite(intensity) or not -3 <= intensity <= 3:
            raise ValueError(f"invalid intensity at sample {row['sample_no']}")
        if (not all(math.isfinite(value) and 0 <= value <= 1 for value in probability)
                or abs(sum(probability) - 1) > 1e-5):
            raise ValueError(f"invalid probability at sample {row['sample_no']}")
        predicted = ("Negative", "Neutral", "Positive")[max(
            range(3), key=probability.__getitem__)]
        if row["pred_label"] != predicted:
            raise ValueError(f"label/probability mismatch at sample {row['sample_no']}")
    print(f"validated {len(rows)} attachment-3 predictions: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=Path)
    args = parser.parse_args()
    validate(args.predictions_csv)
