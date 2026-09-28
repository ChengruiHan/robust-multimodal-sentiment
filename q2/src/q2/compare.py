"""Paired validation comparison from cached prediction CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run import macro_f1


def read_predictions(path: Path):
    data = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["condition_id"], {})[row["sample_id"]] = row
    return data


def summary(rows: dict, ids: list[str], indices: np.ndarray):
    target_cls = np.array([int(rows[k]["true_cls"]) for k in ids])[indices]
    pred_cls = np.array([int(rows[k]["pred_cls"]) for k in ids])[indices]
    target_reg = np.array([float(rows[k]["true_reg"]) for k in ids])[indices]
    pred_reg = np.array([float(rows[k]["pred_reg"]) for k in ids])[indices]
    return np.array([macro_f1(target_cls, pred_cls), np.mean(np.abs(target_reg - pred_reg))])


def compare(left: dict, right: dict, selected: list[str], repeats=1000, seed=3407):
    if set(left) != set(right):
        raise ValueError("condition mismatch")
    ids = sorted(left["clean"])
    if any(set(left[c]) != set(ids) or set(right[c]) != set(ids) for c in selected):
        raise ValueError("sample IDs differ across models or conditions")
    n = len(ids)
    all_indices = np.arange(n)
    rng = np.random.default_rng(seed)
    output = {}
    for condition in selected:
        base = summary(left[condition], ids, all_indices)
        robust = summary(right[condition], ids, all_indices)
        bootstrap = np.empty((repeats, 2))
        for b in range(repeats):
            sample = rng.integers(n, size=n)
            bootstrap[b] = summary(right[condition], ids, sample) - summary(left[condition], ids, sample)
        output[condition] = dict(M1=dict(macro_f1=float(base[0]), mae=float(base[1])),
                                 M5=dict(macro_f1=float(robust[0]), mae=float(robust[1])),
                                 delta_M5_minus_M1=dict(macro_f1=float(robust[0] - base[0]),
                                                        mae=float(robust[1] - base[1])),
                                 ci95=dict(macro_f1=np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
                                           mae=np.quantile(bootstrap[:, 1], [.025, .975]).tolist()))
    return output


def compare_average_missing(left: dict, right: dict, repeats=1000, seed=3407):
    """Resample samples once per replicate, then average the 54 condition scores."""
    conditions = sorted(set(left) - {"clean"})
    ids = sorted(left["clean"])
    if set(left) != set(right) or len(conditions) != 54:
        raise ValueError("expected matching 55-condition grids")
    n = len(ids)
    rng = np.random.default_rng(seed)
    samples = rng.integers(n, size=(repeats, n))
    scores = np.zeros((repeats, 2), dtype=float)
    point = np.zeros(2, dtype=float)
    for condition in conditions:
        if set(left[condition]) != set(ids) or set(right[condition]) != set(ids):
            raise ValueError(f"sample mismatch in {condition}")
        y = np.array([int(left[condition][k]["true_cls"]) for k in ids])
        yl = np.array([float(left[condition][k]["true_reg"]) for k in ids])
        a = np.array([int(left[condition][k]["pred_cls"]) for k in ids])
        b = np.array([int(right[condition][k]["pred_cls"]) for k in ids])
        ar = np.array([float(left[condition][k]["pred_reg"]) for k in ids])
        br = np.array([float(right[condition][k]["pred_reg"]) for k in ids])
        point += (macro_f1(y, b) - macro_f1(y, a),
                  np.mean(np.abs(yl - br) - np.abs(yl - ar)))
        resampled_y, resampled_a, resampled_b = y[samples], a[samples], b[samples]
        for label in range(3):
            def f1(pred):
                tp = ((resampled_y == label) & (pred == label)).sum(1)
                fp = ((resampled_y != label) & (pred == label)).sum(1)
                fn = ((resampled_y == label) & (pred != label)).sum(1)
                return 2 * tp / np.maximum(1, 2 * tp + fp + fn)
            scores[:, 0] += (f1(resampled_b) - f1(resampled_a)) / 3
        scores[:, 1] += (np.abs(yl[samples] - br[samples]) - np.abs(yl[samples] - ar[samples])).mean(1)
    scores /= len(conditions)
    point /= len(conditions)
    return dict(delta_M5_minus_M1=dict(macro_f1=float(point[0]), mae=float(point[1])),
                ci95=dict(macro_f1=np.quantile(scores[:, 0], [.025, .975]).tolist(),
                          mae=np.quantile(scores[:, 1], [.025, .975]).tolist()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m1", required=True)
    p.add_argument("--m5", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    left, right = read_predictions(Path(a.m1)), read_predictions(Path(a.m5))
    selected = ["clean", "T_25_middle", "T_40_end", "A_25_middle", "V_25_middle", "A+V_25_middle"]
    result = compare(left, right, selected)
    result["average_missing"] = compare_average_missing(left, right)
    Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
