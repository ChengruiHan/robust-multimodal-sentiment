"""CPU re-scoring of frozen numerical records, using only the standard library."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "macro_f1", "mae", "pearson")
PROBABILITIES = ("prob_negative", "prob_neutral", "prob_positive")
CONDITIONS = {"clean"} | {
    f"{subset}_{ratio}_{position}"
    for subset in ("T", "A", "V", "T+A", "T+V", "A+V")
    for ratio in (10, 25, 40)
    for position in ("front", "middle", "end")
}


def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inputs(root: Path = ROOT) -> int:
    manifest = json.loads((root / "INPUTS.sha256.json").read_text())
    entries = manifest["files"]
    if not entries:
        raise ValueError("empty evidence manifest")
    for name, expected in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"manifest path escapes repository: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return len(entries)


def score(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty prediction group")
    confusion = [[0] * 3 for _ in range(3)]
    ids, actual, predicted = set(), [], []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in ids:
            raise ValueError(f"duplicate sample: {sample_id}")
        ids.add(sample_id)
        y, yp = int(row["true_cls"]), int(row["pred_cls"])
        z, zp = float(row["true_reg"]), float(row["pred_reg"])
        probs = [float(row[k]) for k in PROBABILITIES]
        if y not in range(3) or yp not in range(3):
            raise ValueError(f"invalid class: {sample_id}")
        if not all(math.isfinite(x) for x in [z, zp, *probs]):
            raise ValueError(f"nonfinite prediction: {sample_id}")
        if not (-3 <= z <= 3 and -3 <= zp <= 3):
            raise ValueError(f"invalid intensity: {sample_id}")
        if any(p < 0 or p > 1 for p in probs) or abs(sum(probs) - 1) > 1e-6:
            raise ValueError(f"invalid probabilities: {sample_id}")
        if yp != max(range(3), key=probs.__getitem__):
            raise ValueError(f"class differs from probability argmax: {sample_id}")
        confusion[y][yp] += 1
        actual.append(z)
        predicted.append(zp)
    n = len(rows)
    f1 = []
    for k in range(3):
        denominator = sum(confusion[k]) + sum(row[k] for row in confusion)
        f1.append(2 * confusion[k][k] / denominator if denominator else 0.0)
    a_mean, p_mean = math.fsum(actual) / n, math.fsum(predicted) / n
    a = [x - a_mean for x in actual]
    p = [x - p_mean for x in predicted]
    denominator = math.sqrt(math.fsum(x*x for x in a) * math.fsum(x*x for x in p))
    return dict(n=n, accuracy=sum(confusion[k][k] for k in range(3)) / n,
                macro_f1=math.fsum(f1) / 3,
                mae=math.fsum(abs(x-y) for x, y in zip(actual, predicted)) / n,
                pearson=math.fsum(x*y for x, y in zip(a, p)) / denominator if denominator else None,
                confusion=confusion)


def compare_metrics(actual: dict, recorded: dict, tolerance: float, label: str) -> None:
    for key in METRICS:
        if actual[key] is None or not math.isclose(actual[key], float(recorded[key]),
                                                  rel_tol=0, abs_tol=tolerance):
            raise ValueError(f"{label}: {key} differs from the recorded value")


def rescore(root: Path = ROOT) -> dict:
    verified = verify_inputs(root)
    rows = read_csv(root / "results/state_msa/valid_predictions.csv.gz")
    groups = {}
    for row in rows:
        groups.setdefault(row["condition_id"], []).append(row)
    if set(groups) != CONDITIONS:
        raise ValueError("expected clean and exactly the 54 specified missing conditions")
    truth = {r["sample_id"]: (r["true_cls"], r["true_reg"]) for r in groups["clean"]}
    if len(truth) != 728:
        raise ValueError("expected 728 validation samples")
    scores = {}
    for condition, records in groups.items():
        targets = {r["sample_id"]: (r["true_cls"], r["true_reg"]) for r in records}
        if targets != truth:
            raise ValueError(f"sample IDs or targets differ: {condition}")
        scores[condition] = score(records)
    metric_rows = read_csv(root / "results/state_msa/valid_55_conditions.csv")
    if len(metric_rows) != 55 or {r["condition_id"] for r in metric_rows} != CONDITIONS:
        raise ValueError("recorded metric grid is incomplete or duplicated")
    for row in metric_rows:
        compare_metrics(scores[row["condition_id"]], row, 1e-8, row["condition_id"])
    missing = [v for k, v in scores.items() if k != "clean"]
    missing_mean = {k: math.fsum(s[k] for s in missing) / 54 for k in METRICS}
    trace_rows = read_csv(root / "results/trace_msa/valid_predictions.csv")
    trace_targets = {r["sample_id"]: (r["true_cls"], r["true_reg"]) for r in trace_rows}
    if trace_targets != truth:
        raise ValueError("TRACE-MSA sample IDs or targets differ")
    trace = score(trace_rows)
    recorded_trace = json.loads((root / "results/trace_msa/valid_metrics.json").read_text())
    compare_metrics(trace, recorded_trace, 1e-6, "TRACE-MSA")
    if trace["confusion"] != recorded_trace["confusion"] or trace["n"] != recorded_trace["n"]:
        raise ValueError("TRACE-MSA confusion or sample count differs")
    fidelity = read_csv(root / "results/trace_msa/valid_fidelity.csv")
    if len(fidelity) != 728 or {r["sample_id"] for r in fidelity} != set(truth):
        raise ValueError("fidelity must contain each validation sample exactly once")
    deltas = []
    for row in fidelity:
        aopc, random, delta = (float(row[k]) for k in ("aopc", "random_aopc", "delta_aopc"))
        if not all(math.isfinite(v) for v in (aopc, random, delta)) or not math.isclose(
                aopc - random, delta, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"invalid fidelity arithmetic: {row['sample_id']}")
        deltas.append(delta)
    mean_delta = math.fsum(deltas) / len(deltas)
    if not math.isclose(mean_delta, recorded_trace["fidelity"]["mean_delta_aopc"], abs_tol=1e-8):
        raise ValueError("recorded fidelity mean differs")
    return dict(verified_files=verified, split="valid", conditions=55,
                prediction_rows=len(rows), samples_per_condition=728,
                state_msa=dict(clean=scores["clean"], missing_mean=missing_mean,
                               by_condition=scores),
                trace_msa=dict(clean=trace, fidelity_mean_delta_aopc=mean_delta),
                scope="Numerical re-scoring of saved predictions; no model inference. "
                      "Bootstrap intervals and width-comparison aggregates are not regenerated.")
