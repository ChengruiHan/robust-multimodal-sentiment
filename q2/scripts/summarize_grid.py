"""Summarize a fixed 55-condition evaluation without changing predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 55 or len({row["condition_id"] for row in rows}) != 55:
        raise ValueError("expected one clean and 54 distinct missing conditions")
    clean = next(row for row in rows if row["condition_id"] == "clean")
    missing = [row for row in rows if row["condition_id"] != "clean"]
    keys = ("accuracy", "macro_f1", "mae", "pearson")
    return {
        "clean": {key: float(clean[key]) for key in keys},
        "missing_mean": {
            key: sum(float(row[key]) for row in missing) / 54 for key in keys
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.metrics_csv), indent=2, ensure_ascii=False))
