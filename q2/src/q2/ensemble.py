"""Equal-weight ensemble of frozen prediction grids from independent seeds."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .compare import read_predictions
from .run import macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grids = [read_predictions(Path(path)) for path in args.inputs]
    if len(grids) < 2 or any(set(grid) != set(grids[0]) for grid in grids[1:]):
        raise ValueError("need matching grids from at least two seeds")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition_id", "sample_id", "true_cls", "true_reg", "pred_cls",
                         "pred_reg", "prob_negative", "prob_neutral", "prob_positive"))
        for condition in sorted(grids[0]):
            ids = sorted(grids[0][condition])
            if any(set(grid[condition]) != set(ids) for grid in grids[1:]):
                raise ValueError(f"sample mismatch in {condition}")
            target_cls, target_reg, probs, scores = [], [], [], []
            for sample_id in ids:
                rows = [grid[condition][sample_id] for grid in grids]
                labels = {(row["true_cls"], row["true_reg"]) for row in rows}
                if len(labels) != 1:
                    raise ValueError(f"target mismatch for {sample_id}")
                probability = np.mean([[float(row[f"prob_{name}"]) for name in
                                        ("negative", "neutral", "positive")] for row in rows], axis=0)
                score = float(np.mean([float(row["pred_reg"]) for row in rows]))
                truth_cls, truth_reg = int(rows[0]["true_cls"]), float(rows[0]["true_reg"])
                writer.writerow((condition, sample_id, truth_cls, truth_reg,
                                 int(probability.argmax()), score, *map(float, probability)))
                target_cls.append(truth_cls)
                target_reg.append(truth_reg)
                probs.append(probability)
                scores.append(score)
            y = np.asarray(target_cls)
            z = np.asarray(target_reg)
            predicted = np.asarray(probs).argmax(1)
            scores = np.asarray(scores)
            metrics.append(dict(condition_id=condition, accuracy=float(np.mean(y == predicted)),
                                macro_f1=macro_f1(y, predicted), mae=float(np.mean(np.abs(z - scores))),
                                pearson=float(np.corrcoef(z, scores)[0, 1])))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(next((row for row in metrics if row["condition_id"] == "clean"),
               {"conditions": len(metrics), "output": str(output)}))


if __name__ == "__main__":
    main()
