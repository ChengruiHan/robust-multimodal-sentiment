"""Publication-ready plots from frozen Q2 validation caches."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODALITIES = ("T", "A", "V", "T+A", "T+V", "A+V")
COLORS = dict(zip(MODALITIES, plt.cm.tab10.colors[:6]))


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--extra")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    grid = read_csv(Path(a.grid) / "metrics.csv")
    preds = [r for r in read_csv(Path(a.grid) / "predictions.csv") if r["condition_id"] == "clean"]
    extra = ({r["condition_id"]: r for r in read_csv(Path(a.extra) / "metrics.csv")}
             if a.extra else {})
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)

    y = np.array([int(r["true_cls"]) for r in preds])
    yp = np.array([int(r["pred_cls"]) for r in preds])
    cm = np.array([[((y == i) & (yp == j)).sum() for j in range(3)] for i in range(3)])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=("Negative", "Neutral", "Positive"),
           yticklabels=("Negative", "Neutral", "Positive"), xlabel="Predicted", ylabel="True",
           title="Clean validation confusion matrix")
    fig.colorbar(image, ax=ax)
    save(fig, output / "confusion_matrix.png")

    values = defaultdict(dict)
    for r in grid:
        if r["condition_id"] == "clean":
            continue
        name, ratio, position = r["condition_id"].rsplit("_", 2)
        values[(name, int(ratio))][position] = r
    for metric, filename, ylabel in (("macro_f1", "ratio_f1.png", "Macro-F1"),
                                      ("mae", "ratio_mae.png", "MAE")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in MODALITIES:
            ratios = [10, 25, 40]
            means = [np.mean([float(v[metric]) for v in values[(name, r)].values()]) for r in ratios]
            ax.plot(ratios, means, marker="o", label=name, color=COLORS[name])
        ax.set(xlabel="Missing ratio (%)", ylabel=ylabel, xticks=[10, 25, 40],
               title=f"Validation {ylabel} by missing ratio")
        ax.grid(alpha=.25)
        ax.legend(ncol=3)
        save(fig, output / filename)

    matrix = np.array([[float(values[(name, 25)][position]["macro_f1"])
                        for position in ("front", "middle", "end")] for name in MODALITIES])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=matrix.min(), vmax=matrix.max())
    ax.set(xticks=range(3), yticks=range(6), xticklabels=("Front", "Middle", "End"),
           yticklabels=MODALITIES, title="Macro-F1 at 25% missing")
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax)
    save(fig, output / "position_heatmap.png")

    if extra:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in ("T", "A", "V"):
            gaps = [1, 2, 3, 5, 8]
            ax.plot(gaps, [float(extra[f"gap_{name}_{g}"]["mae"]) for g in gaps],
                    marker="o", label=name, color=COLORS[name])
        ax.set(xlabel="Contiguous missing length (aligned positions)", ylabel="MAE",
               title="Single middle gap")
        ax.grid(alpha=.25)
        ax.legend()
        save(fig, output / "gap_length.png")

    yr = np.array([float(r["true_reg"]) for r in preds])
    ypr = np.array([float(r["pred_reg"]) for r in preds])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yr, ypr, s=10, alpha=.4)
    ax.plot([-3, 3], [-3, 3], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="True intensity", ylabel="Predicted intensity", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1),
           title="Clean validation regression")
    save(fig, output / "regression_scatter.png")


if __name__ == "__main__":
    main()
