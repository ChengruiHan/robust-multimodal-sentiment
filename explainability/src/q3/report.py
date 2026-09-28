"""Render Q3 validation figures and evidence-backed error summary from run outputs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def report(input_dir: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((input_dir / "valid_metrics.json").read_text())
    predictions = rows(input_dir / "valid_predictions_explanations.csv")
    cards = json.loads((input_dir / "valid_explanation_cards.json").read_text())
    matrix = np.asarray(metrics["confusion"], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xticks(range(3), ("Negative", "Neutral", "Positive"))
    ax.set_yticks(range(3), ("Negative", "Neutral", "Positive"))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "valid_confusion.png", dpi=180)
    plt.close(fig)

    truth = np.array([c["true_reg"] for c in cards], dtype=float)
    estimate = np.array([c["prediction"]["pred_intensity"] for c in cards], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(truth, estimate, s=10, alpha=.4)
    ax.plot([-3, 3], [-3, 3], color="gray", linestyle="--")
    ax.set(xlabel="True intensity", ylabel="Predicted intensity", xlim=(-3, 3), ylim=(-3, 3))
    fig.tight_layout()
    fig.savefig(output_dir / "valid_intensity.png", dpi=180)
    plt.close(fig)

    signed = np.array([[float(row[f"phi_{m}_cls"]) for m in ("text", "audio", "vision")]
                       for row in predictions])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for m, ax in enumerate(axes):
        ax.hist(signed[:, m], bins=35, color=("#4c78a8", "#f58518", "#54a24b")[m])
        ax.axvline(0, color="black", linewidth=.7)
        ax.set_title(("Text", "Audio", "Vision")[m])
        ax.set_xlabel("Signed Shapley contribution")
    axes[0].set_ylabel("Validation samples")
    fig.tight_layout()
    fig.savefig(output_dir / "valid_modality_contributions.png", dpi=180)
    plt.close(fig)

    primary = Counter(row["primary_modality"] for row in predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ("text", "audio", "vision", "undetermined")
    ax.bar(labels, [primary[x] for x in labels])
    ax.set_ylabel("Validation samples")
    fig.tight_layout()
    fig.savefig(output_dir / "valid_primary_modality.png", dpi=180)
    plt.close(fig)

    by_label = np.zeros((3, 3), dtype=int)
    for row in predictions:
        if row["primary_modality"] in ("text", "audio", "vision"):
            label = ("Negative", "Neutral", "Positive").index(row["pred_label"])
            modality = ("text", "audio", "vision").index(row["primary_modality"])
            by_label[label, modality] += 1
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(3, dtype=int)
    for m, name in enumerate(("text", "audio", "vision")):
        ax.bar(("Negative", "Neutral", "Positive"), by_label[:, m], bottom=bottom,
               label=name)
        bottom += by_label[:, m]
    ax.set_ylabel("Validation samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "valid_primary_by_prediction.png", dpi=180)
    plt.close(fig)

    fidelity_path = input_dir / "valid_fidelity.csv"
    if fidelity_path.exists():
        faith = rows(fidelity_path)
        selected = np.array([float(row["aopc"]) for row in faith if row["aopc"]], dtype=float)
        random = np.array([float(row["random_aopc"]) for row in faith if row["random_aopc"]], dtype=float)
        if len(selected) and len(random):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.boxplot([selected, random], tick_labels=["Top evidence", "Random matched"])
            ax.set_ylabel("AOPC")
            fig.tight_layout()
            fig.savefig(output_dir / "valid_fidelity.png", dpi=180)
            plt.close(fig)

    by_id = {row["sample_id"]: row for row in predictions}
    case_counts = Counter()
    examples = {}
    group = {"correct": [], "error": []}
    for card in cards:
        row = by_id[card["sample_id"]]
        predicted = ("Negative", "Neutral", "Positive")[card["true_cls"]]
        correct = row["pred_label"] == predicted
        phi = np.array([float(row[f"phi_{m}_cls"]) for m in ("text", "audio", "vision")])
        group["correct" if correct else "error"].append(dict(
            conflict=bool((phi > 1e-8).any() and (phi < -1e-8).any()),
            text_primary=row["primary_modality"] == "text",
            weak=row["weak_explanation"] == "1",
            neutral_related=card["true_cls"] == 1 or row["pred_label"] == "Neutral",
            margin=float(row["prediction_margin"]),
            intensity_abs_error=abs(float(row["pred_intensity"]) - float(card["true_reg"]))))
        if correct:
            continue
        cases = []
        if row["primary_modality"] == "text":
            cases.append("text_primary_error")
        if (phi > 0).any() and (phi < 0).any():
            cases.append("cross_modal_conflict_error")
        if row["weak_explanation"] == "1":
            cases.append("weak_explanation_error")
        if card["true_cls"] == 1 or row["pred_label"] == "Neutral":
            cases.append("neutral_related_error")
        for kind in cases:
            case_counts[kind] += 1
            examples.setdefault(kind, []).append(card["sample_id"])
    rates = {name: {key: float(np.mean([item[key] for item in values])) if values else None
                    for key in ("conflict", "text_primary", "weak", "neutral_related",
                                "margin", "intensity_abs_error")}
             for name, values in group.items()}
    summary = dict(n_validation=len(predictions), n_errors=sum(
        card["prediction"]["pred_label"] != ("Negative", "Neutral", "Positive")[card["true_cls"]]
        for card in cards), counts=dict(case_counts), example_ids={k: v[:10] for k, v in examples.items()},
        group_means=rates,
        interpretation="Diagnostic associations; these categories do not establish causes of error.")
    (output_dir / "valid_error_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def attachment_cards(input_dir: Path, output_dir: Path) -> None:
    """Render one model-position evidence plot per attachment-4 sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    cards = json.loads((input_dir / "attachment4_explanation_cards.json").read_text())
    position_rows = rows(input_dir / "attachment4_position_importance.csv")
    shares = np.array([[float(card["prediction"][f"share_{m}"])
                        for m in ("text", "audio", "vision")] for card in cards])
    fig, ax = plt.subplots(figsize=(10, 4))
    bottoms = np.zeros(len(cards))
    for m, name in enumerate(("text", "audio", "vision")):
        ax.bar([card["sample_id"] for card in cards], shares[:, m], bottom=bottoms,
               label=name)
        bottoms += shares[:, m]
    ax.set(xlabel="Attachment 4 sample", ylabel="Absolute Shapley share", ylim=(0, 1))
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "attachment4_modality_comparison.png", dpi=180)
    plt.close(fig)
    by_id = {}
    for row in position_rows:
        by_id.setdefault(row["sample_id"], []).append(row)
    for card in cards:
        prediction = card["prediction"]
        sample_id = card["sample_id"]
        primary = prediction["primary_modality"]
        fig, ax = plt.subplots(figsize=(10, 4))
        for modality, color in (("text", "#4c78a8"), ("audio", "#f58518"),
                                ("vision", "#54a24b")):
            series = [row for row in by_id.get(sample_id, []) if row["modality"] == modality]
            if not series:
                continue
            x = np.array([int(row["position"]) for row in series])
            y = np.array([float(row["importance"]) for row in series])
            ax.plot(x, y, marker=".", color=color, linewidth=2 if modality == primary else 1,
                    alpha=1 if modality == primary else .5, label=modality)
        for evidence in card["evidence"]:
            if evidence["modality"] == primary and evidence["evidence_type"] == "support" and evidence["rank"] == 1:
                ax.axvspan(evidence["start_idx"], evidence["end_idx"], color="gold", alpha=.22)
                break
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set(xlabel="Aligned position (0-based)", ylabel="Signed local deletion effect",
               title=f"{sample_id}: {prediction['pred_label']} {prediction['pred_intensity']:.2f}; primary={primary}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{sample_id}_importance.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "attachment4"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "validate":
        report(args.input, args.output)
    else:
        attachment_cards(args.input, args.output)


if __name__ == "__main__":
    main()
