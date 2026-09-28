"""Compare fixed width 1/3/5 Q3 validation explanations without retuning them."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODALITIES = ("text", "audio", "vision")
WIDTHS = (1, 3, 5)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample IDs")
    return result


def top_support(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        if row["evidence_type"] == "support" and row["rank"] == "1":
            key = (row["sample_id"], row["modality"])
            if key in result:
                raise ValueError(f"duplicate top support: {key}")
            result[key] = row
    return result


def positions(rows: list[dict]) -> dict[tuple[str, str], dict[int, float]]:
    result = defaultdict(dict)
    for row in rows:
        key = (row["sample_id"], row["modality"])
        pos = int(row["position"])
        if pos in result[key]:
            raise ValueError(f"duplicate position: {key} {pos}")
        result[key][pos] = float(row["importance"])
    return dict(result)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    x, y = average_ranks(np.asarray(left, dtype=float)), average_ranks(np.asarray(right, dtype=float))
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def interval_stats(left: dict, right: dict) -> dict:
    a, b = int(left["start_idx"]), int(left["end_idx"])
    c, d = int(right["start_idx"]), int(right["end_idx"])
    overlap = max(0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    center_distance = abs((a + b - 1) / 2 - (c + d - 1) / 2)
    return dict(overlap=bool(overlap), iou=overlap / union, center_distance=center_distance)


def summarize(rows: list[dict]) -> dict:
    def describe(values):
        arr = np.asarray(values, dtype=float)
        return dict(n=len(arr), mean=float(np.mean(arr)) if len(arr) else None,
                    median=float(np.median(arr)) if len(arr) else None,
                    q25=float(np.quantile(arr, .25)) if len(arr) else None,
                    q75=float(np.quantile(arr, .75)) if len(arr) else None)

    return dict(
        n=len(rows),
        top_support_both=sum(row["top_support_both"] for row in rows),
        overlap_rate=float(np.mean([row["overlap"] for row in rows if row["top_support_both"]]))
        if any(row["top_support_both"] for row in rows) else None,
        center_distance_le2_rate=float(np.mean([row["center_distance"] <= 2 for row in rows
                                             if row["top_support_both"]]))
        if any(row["top_support_both"] for row in rows) else None,
        iou=describe([row["iou"] for row in rows if row["top_support_both"]]),
        center_distance=describe([row["center_distance"] for row in rows if row["top_support_both"]]),
        position_spearman=describe([row["position_spearman"] for row in rows
                                    if row["position_spearman"] is not None]),
    )


def compare(directories: dict[int, Path], output: Path) -> dict:
    predictions = {w: keyed(read_csv(directories[w] / "valid_predictions_explanations.csv")) for w in WIDTHS}
    evidences = {w: top_support(read_csv(directories[w] / "valid_evidence_detail.csv")) for w in WIDTHS}
    curves = {w: positions(read_csv(directories[w] / "valid_position_importance.csv")) for w in WIDTHS}
    metrics = {w: json.loads((directories[w] / "valid_metrics.json").read_text()) for w in WIDTHS}
    ids = list(predictions[3])
    if any(set(predictions[w]) != set(ids) for w in WIDTHS):
        raise ValueError("validation ID sets differ across widths")
    model_ids = {row["model_id"] for width in WIDTHS for row in predictions[width].values()}
    if len(model_ids) != 1:
        raise ValueError(f"model IDs differ: {model_ids}")
    max_differences = {}
    for w in (1, 5):
        if any(predictions[w][sid]["pred_label"] != predictions[3][sid]["pred_label"] for sid in ids):
            raise AssertionError(f"width {w} changed a predicted class")
        if any(predictions[w][sid]["primary_modality"] != predictions[3][sid]["primary_modality"]
               or predictions[w][sid]["primary_sign"] != predictions[3][sid]["primary_sign"]
               for sid in ids):
            raise AssertionError(f"width {w} changed a modality explanation")
        max_differences[str(w)] = {field: max(abs(float(predictions[w][sid][field]) -
                                                  float(predictions[3][sid][field])) for sid in ids)
                                   for field in ("pred_intensity", "prob_negative", "prob_neutral",
                                                 "prob_positive", "phi_text_cls", "phi_audio_cls",
                                                 "phi_vision_cls", "share_text", "share_audio",
                                                 "share_vision")}
    comparison = []
    for sid in ids:
        primary = predictions[3][sid]["primary_modality"]
        for modality in MODALITIES:
            for width in (1, 5):
                pair = (sid, modality)
                base, other = evidences[3].get(pair), evidences[width].get(pair)
                a, b = curves[3].get(pair, {}), curves[width].get(pair, {})
                common = sorted(set(a) & set(b))
                span = interval_stats(base, other) if base is not None and other is not None else None
                comparison.append(dict(sample_id=sid, modality=modality,
                                       primary_modality=(modality == primary), width=width,
                                       positions_3=len(a), positions_other=len(b),
                                       shared_positions=len(common),
                                       top_support_both=span is not None,
                                       overlap=span["overlap"] if span else None,
                                       iou=span["iou"] if span else None,
                                       center_distance=span["center_distance"] if span else None,
                                       position_spearman=spearman([a[t] for t in common],
                                                                  [b[t] for t in common]),
                                       start_3=int(base["start_idx"]) if base else None,
                                       end_3=int(base["end_idx"]) if base else None,
                                       start_other=int(other["start_idx"]) if other else None,
                                       end_other=int(other["end_idx"]) if other else None,
                                       delta_3=float(base["delta_cls"]) if base else None,
                                       delta_other=float(other["delta_cls"]) if other else None))
    summary = dict(model_id=model_ids.pop(), n_samples=len(ids), widths=list(WIDTHS),
                   prediction_invariance_max_abs_diff=max_differences,
                   metrics={str(w): {key: metrics[w][key] for key in ("accuracy", "macro_f1", "mae", "pearson")}
                            for w in WIDTHS},
                   groups={})
    for width in (1, 5):
        subset = [row for row in comparison if row["width"] == width]
        summary["groups"][str(width)] = dict(
            all_modalities=summarize(subset),
            primary_modality=summarize([row for row in subset if row["primary_modality"]]),
            by_modality={m: summarize([row for row in subset if row["modality"] == m])
                         for m in MODALITIES})
    if all((directories[w] / "valid_fidelity.csv").exists() for w in WIDTHS):
        fidelity = {w: keyed(read_csv(directories[w] / "valid_fidelity.csv")) for w in WIDTHS}
        if any(set(fidelity[w]) != set(ids) for w in WIDTHS):
            raise ValueError("fidelity ID sets differ across widths")
        summary["fidelity"] = {}
        differences = {}
        for w in WIDTHS:
            valid_ids = [sid for sid in ids if fidelity[w][sid]["delta_aopc"]]
            values = np.array([float(fidelity[w][sid]["delta_aopc"]) for sid in valid_ids])
            top = np.array([float(fidelity[w][sid]["aopc"]) for sid in valid_ids])
            random = np.array([float(fidelity[w][sid]["random_aopc"]) for sid in valid_ids])
            differences[w] = {sid: value for sid, value in zip(valid_ids, values)}
            summary["fidelity"][str(w)] = dict(n=len(valid_ids), mean_top=float(top.mean()),
                                               mean_random=float(random.mean()),
                                               mean_delta=float(values.mean()),
                                               positive_count=int((values > 0).sum()))
        rng = np.random.default_rng(3407)
        summary["fidelity_paired_delta_difference"] = {}
        for w in (1, 5):
            paired_ids = [sid for sid in ids if sid in differences[w] and sid in differences[3]]
            paired = np.array([differences[w][sid] - differences[3][sid] for sid in paired_ids])
            samples = rng.choice(paired, size=(2000, len(paired)), replace=True).mean(axis=1)
            summary["fidelity_paired_delta_difference"][f"{w}_minus_3"] = dict(
                n=len(paired), mean=float(paired.mean()),
                ci95=np.quantile(samples, [.025, .975]).tolist())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "width_sensitivity_per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (output / "width_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_figure(summary, output / "width_sensitivity.png")
    return summary


def render_figure(summary: dict, target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("Text", "Audio", "Vision")
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, measure, ylabel in (
        (axes[0], "overlap_rate", "Top support span overlap rate"),
        (axes[1], "position_spearman", "Median position-rank Spearman"),
    ):
        for offset, width, color in ((-.17, 1, "#4c78a8"), (.17, 5, "#f58518")):
            values = []
            for modality in MODALITIES:
                item = summary["groups"][str(width)]["by_modality"][modality]
                values.append(item[measure] if measure == "overlap_rate"
                              else item[measure]["median"])
            axis.bar(x + offset, [float(v) if v is not None else np.nan for v in values],
                     width=.32, color=color, label=f"width {width} vs 3")
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
    fig.suptitle(f"Q3 window sensitivity on {summary['n_samples']} validation samples")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for width in WIDTHS:
        parser.add_argument(f"--width-{width}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare({w: getattr(args, f"width_{w}") for w in WIDTHS}, args.output), indent=2))


if __name__ == "__main__":
    main()
