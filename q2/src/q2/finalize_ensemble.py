"""Aggregate three frozen Q2 model outputs without fitting on test or attachment 3."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run import macro_f1


LABELS = ("Negative", "Neutral", "Positive")
PROBABILITY_COLUMNS = ("prob_negative", "prob_neutral", "prob_positive")


def read_by(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {row[key]: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"duplicate {key} in {path}")
    return by_key


def average(rows: list[dict[str, str]]) -> tuple[np.ndarray, float]:
    probability = np.mean([[float(row[name]) for name in PROBABILITY_COLUMNS]
                           for row in rows], axis=0)
    score_column = "pred_reg" if "pred_reg" in rows[0] else "pred_intensity"
    score = float(np.mean([float(row[score_column]) for row in rows]))
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1, atol=1e-5):
        raise ValueError("invalid ensemble probabilities")
    if not np.isfinite(score) or not -3 <= score <= 3:
        raise ValueError("invalid ensemble intensity")
    return probability, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "attachment3"), required=True)
    parser.add_argument("--inputs", nargs=3, required=True)
    parser.add_argument("--diagnostics", help="First model's attachment-3 diagnostics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    key = "sample_id" if args.mode == "test" else "sample_no"
    records = [read_by(Path(path), key) for path in args.inputs]
    ids = sorted(records[0], key=int if args.mode == "attachment3" else None)
    if any(set(record) != set(ids) for record in records[1:]):
        raise ValueError("sample IDs differ across ensemble members")
    if args.mode == "attachment3" and [int(value) for value in ids] != list(range(1, 31)):
        raise ValueError("attachment 3 must contain samples 1..30")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    filename = "clean_test_predictions.csv" if args.mode == "test" else "attachment3_predictions.csv"
    target = output / filename
    if target.exists():
        raise FileExistsError(target)
    predictions = []
    for sample_id in ids:
        rows = [record[sample_id] for record in records]
        probability, score = average(rows)
        if args.mode == "test":
            truths = {(row["true_cls"], row["true_reg"]) for row in rows}
            if len(truths) != 1:
                raise ValueError(f"target mismatch for {sample_id}")
            predictions.append(dict(sample_id=sample_id, true_cls=int(rows[0]["true_cls"]),
                                    true_reg=float(rows[0]["true_reg"]),
                                    pred_cls=int(probability.argmax()), pred_reg=score,
                                    **dict(zip(PROBABILITY_COLUMNS, map(float, probability)))))
        else:
            predictions.append(dict(sample_no=int(sample_id), pred_label=LABELS[int(probability.argmax())],
                                    pred_intensity=score,
                                    **dict(zip(PROBABILITY_COLUMNS, map(float, probability)))))
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    if args.mode == "test":
        y = np.array([row["true_cls"] for row in predictions])
        z = np.array([row["true_reg"] for row in predictions])
        yp = np.array([row["pred_cls"] for row in predictions])
        zp = np.array([row["pred_reg"] for row in predictions])
        report = dict(split="test", condition="clean", n=len(y), models=len(records),
                      metrics=dict(accuracy=float(np.mean(y == yp)), macro_f1=macro_f1(y, yp),
                                   mae=float(np.mean(np.abs(z - zp))),
                                   pearson=float(np.corrcoef(z, zp)[0, 1])))
        (output / "clean_test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(json.dumps(report, indent=2))
    else:
        if not args.diagnostics:
            raise ValueError("attachment3 requires diagnostics")
        diagnostics = read_by(Path(args.diagnostics), "sample_no")
        if set(diagnostics) != set(ids):
            raise ValueError("diagnostic IDs differ")
        adjusted = []
        for row in predictions:
            detail = dict(diagnostics[str(row["sample_no"])])
            probability = np.array([row[name] for name in PROBABILITY_COLUMNS])
            detail["pred_entropy"] = float(-(probability * np.log(np.clip(probability, 1e-12, 1))).sum())
            detail["max_probability"] = float(probability.max())
            adjusted.append(detail)
        with (output / "attachment3_missing_diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(adjusted[0]))
            writer.writeheader()
            writer.writerows(adjusted)
        print(f"wrote {len(predictions)} attachment-3 ensemble predictions")


if __name__ == "__main__":
    main()
