"""TRACE-MSA command line: validate on attachment 2 and infer attachment 4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import subprocess

import numpy as np

from .core import (Ensemble, MODALITIES, explain_modalities, explain_windows,
                   position_importance, require_sample, score_class, select_evidence,
                   slice_sample)
from .trace import build_trace, locate

LABELS = ("Negative", "Neutral", "Positive")
MAIN_COLUMNS = ("sample_id", "model_id", "pred_label", "pred_intensity", "prob_negative",
                "prob_neutral", "prob_positive", "prediction_margin", "phi_text_cls",
                "phi_audio_cls", "phi_vision_cls", "share_text", "share_audio", "share_vision",
                "primary_modality", "primary_sign", "main_support_modality", "phi_text_reg",
                "phi_audio_reg", "phi_vision_reg", "weak_explanation", "raw_unavailable",
                "coalition_support_warning", "localization_quality", "config_sha256")
EVIDENCE_COLUMNS = ("sample_id", "model_id", "modality", "evidence_type", "rank",
                    "start_idx", "end_idx", "delta_cls", "delta_reg", "text_char_start",
                    "text_char_end", "text_span", "start_sec", "end_sec", "keyframe_sec",
                    "frame_path", "mapping_quality", "mapping_warning")


def extract_frame(video: Path, second: float, target: Path) -> str | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{second:.6f}",
               "-i", str(video), "-frames:v", "1", "-q:v", "4", str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    return str(Path("keyframes") / target.name) if result.returncode == 0 and target.is_file() else None


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_attachment(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        data = pickle.load(stream)  # Trusted competition file only.
    required = {"id", "raw_text", "text_bert", "audio", "vision"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"{path.name}: missing attachment-4 fields")
    bert = np.asarray(data["text_bert"], dtype=np.int64)[None]
    audio = np.asarray(data["audio"], dtype=np.float32)[None]
    vision = np.asarray(data["vision"], dtype=np.float32)[None]
    if bert.shape != (1, 3, 50) or audio.shape != (1, 50, 74) or vision.shape != (1, 50, 35):
        raise ValueError(f"{path.name}: wrong aligned_50 shapes")
    ids, attention = bert[0, 0], bert[0, 1]
    if ids[0] != 101 or not np.isin(attention, [0, 1]).all():
        raise ValueError(f"{path.name}: invalid BERT special tokens or attention")
    end = int(attention.sum()) - 1
    if end < 1 or ids[end] != 102:
        raise ValueError(f"{path.name}: invalid [SEP] position")
    content = (attention.astype(bool) & (ids != 101) & (ids != 102))[None]
    raw = np.stack((content, content & np.any(audio != 0, 2),
                    content & np.any(vision != 0, 2)), axis=1)
    sample = dict(id=np.asarray([str(data["id"])]), text_bert=bert, audio=audio,
                  vision=vision, content=content, raw_available=raw)
    require_sample(sample)
    if str(data["id"]) != path.stem:
        raise ValueError(f"{path.name}: sample ID mismatch")
    return sample, str(data["raw_text"])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_id(paths: list[Path], bert: Path) -> str:
    hashes = []
    for path in paths:
        hashes.append(file_hash(path))
        hashes.append(file_hash(path.parent / "scaler.npz"))
    weights = bert / "model.safetensors"
    if weights.is_file():
        hashes.append(file_hash(weights))
    else:
        raise FileNotFoundError(f"frozen BERT weight not found: {weights}")
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]


def main_row(sample_id: str, model: str, modal: dict, config_hash: str,
             localization_quality: str) -> dict:
    probability = np.asarray(modal["probabilities"], dtype=float)
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1, atol=1e-5):
        raise ValueError(f"{sample_id}: invalid probabilities")
    intensity = float(modal["intensity"])
    if not -3.00001 <= intensity <= 3.00001:
        raise ValueError(f"{sample_id}: invalid intensity")
    phi, reg = modal["phi_cls"], modal["phi_reg"]
    shares = modal["summary"]["shares"]
    if shares[0] is not None and not np.isclose(sum(shares), 1, atol=1e-7):
        raise ValueError(f"{sample_id}: modality shares do not sum to one")
    top = np.sort(probability)[-2:]
    return dict(sample_id=sample_id, model_id=model,
                pred_label=LABELS[modal["target"]], pred_intensity=intensity,
                prob_negative=float(probability[0]), prob_neutral=float(probability[1]),
                prob_positive=float(probability[2]), prediction_margin=float(top[1] - top[0]),
                phi_text_cls=float(phi[0]), phi_audio_cls=float(phi[1]),
                phi_vision_cls=float(phi[2]), share_text=shares[0],
                share_audio=shares[1], share_vision=shares[2],
                primary_modality=modal["summary"]["primary"],
                primary_sign=modal["summary"]["primary_sign"],
                main_support_modality=modal["summary"]["main_support"],
                phi_text_reg=float(reg[0]), phi_audio_reg=float(reg[1]),
                phi_vision_reg=float(reg[2]),
                weak_explanation=int(modal["summary"]["weak"]),
                raw_unavailable="+".join(modal["raw_unavailable"]),
                coalition_support_warning=1, localization_quality=localization_quality,
                config_sha256=config_hash)


def explain_one(predictor, sample: dict, width: int) -> tuple[dict, list[dict], np.ndarray]:
    modal = explain_modalities(predictor, sample)
    windows = explain_windows(predictor, sample, modal, width)
    evidence = select_evidence(windows)
    importance = position_importance(windows)
    return modal, evidence, importance


def attachment4(args, predictor, model: str, config_hash: str) -> None:
    root = Path(args.input)
    files = sorted(root.glob("[0-9][0-9].pkl"))
    if [p.stem for p in files] != [f"{i:02d}" for i in range(1, 21)]:
        raise ValueError("attachment 4 must contain exactly 01.pkl through 20.pkl")
    output = Path(args.output)
    rows, details, importance_rows, cards = [], [], [], []
    for path in files:
        sample, raw_text = read_attachment(path)
        modal, evidence, importance = explain_one(predictor, sample, args.window)
        video = root / "videos" / f"{path.stem}.mp4"
        trace, trace_warning = None, ""
        try:
            trace = build_trace(raw_text, sample["text_bert"][0], video, args.bert,
                                align=args.align, device=args.align_device, q1_root=args.q1_root)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            trace_warning = f"trace_failed:{type(exc).__name__}:{exc}"
        sample_evidence = []
        for item in evidence:
            location = locate(trace, item)
            if trace_warning:
                location["mapping_warning"] = trace_warning
            if item["modality"] == "vision" and location["keyframe_sec"] is not None:
                frame = output / "keyframes" / f"{path.stem}_{item['evidence_type']}_{item['rank']}.jpg"
                location["frame_path"] = extract_frame(video, location["keyframe_sec"], frame)
            else:
                location["frame_path"] = None
            row = dict(sample_id=path.stem, model_id=model, **item, **location)
            sample_evidence.append(row)
            details.append(row)
        qualities = [e["mapping_quality"] for e in sample_evidence]
        quality = ("unresolved" if not qualities or "unresolved" in qualities else
                   "approximate" if "approximate" in qualities else "verified")
        rows.append(main_row(path.stem, model, modal, config_hash, quality))
        cards.append(dict(sample_id=path.stem, raw_text=raw_text,
                          prediction=rows[-1], evidence=sample_evidence))
        for m, name in enumerate(MODALITIES):
            for t in np.flatnonzero(sample["raw_available"][0, m]):
                importance_rows.append(dict(sample_id=path.stem, modality=name,
                                            position=int(t), importance=float(importance[m, t])))
        print(f"{path.stem}: {rows[-1]['pred_label']} {rows[-1]['pred_intensity']:.3f} {quality}", flush=True)
    write_csv(output / "attachment4_predictions_explanations.csv", rows, MAIN_COLUMNS)
    write_csv(output / "attachment4_evidence_detail.csv", details, EVIDENCE_COLUMNS)
    write_csv(output / "attachment4_position_importance.csv", importance_rows,
              ("sample_id", "modality", "position", "importance"))
    (output / "attachment4_explanation_cards.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(rows) != 20 or len({r["sample_id"] for r in rows}) != 20:
        raise AssertionError("attachment 4 output cardinality failed")


def scores(actual: np.ndarray, predicted: np.ndarray, true_reg: np.ndarray,
           pred_reg: np.ndarray) -> dict:
    matrix = np.zeros((3, 3), dtype=int)
    for y, p in zip(actual, predicted):
        matrix[int(y), int(p)] += 1
    per_class = []
    for c in range(3):
        tp = matrix[c, c]
        precision = tp / max(1, matrix[:, c].sum())
        recall = tp / max(1, matrix[c].sum())
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class.append(dict(precision=float(precision), recall=float(recall), f1=float(f1)))
    pearson = float(np.corrcoef(true_reg, pred_reg)[0, 1]) if np.std(true_reg) and np.std(pred_reg) else None
    return dict(n=len(actual), accuracy=float(np.mean(actual == predicted)),
                macro_f1=float(np.mean([v["f1"] for v in per_class])),
                mae=float(np.mean(abs(true_reg - pred_reg))), pearson=pearson,
                per_class=per_class, confusion=matrix.tolist())


def fidelity(predictor, sample: dict, modal: dict, evidence: list[dict],
             repeats: int = 50, seed: int = 3407) -> dict:
    """Cumulative supportive deletion versus matched-length random deletion."""
    candidates = sorted((e for e in evidence if e["evidence_type"] == "support"),
                        key=lambda e: -e["delta_cls"])
    support = []
    occupied = np.zeros((3, 50), dtype=bool)
    for item in candidates:
        m, start, end = item["modality_index"], item["start_idx"], item["end_idx"]
        if occupied[m, start:end].any():
            continue
        support.append(item)
        occupied[m, start:end] = True
        if len(support) == 3:
            break
    if not support:
        return dict(aopc=None, random_aopc=None, delta_aopc=None, steps=0)
    rng = random.Random(seed)
    raw = sample["raw_available"][0]
    masks = []
    cumulative = np.zeros((3, 50), dtype=bool)
    for item in support:
        cumulative[item["modality_index"], item["start_idx"]:item["end_idx"]] = True
        masks.append(cumulative.copy())
    random_masks = []
    for _ in range(repeats):
        mask = np.zeros((3, 50), dtype=bool)
        replicate = []
        for item in support:
            m = item["modality_index"]
            length = item["end_idx"] - item["start_idx"]
            starts = [start for start in range(51 - length)
                      if raw[m, start:start + length].all() and not mask[m, start:start + length].any()]
            if not starts:
                replicate = []
                break
            start = rng.choice(starts)
            mask[m, start:start + length] = True
            replicate.append(mask.copy())
        random_masks.extend(replicate)
    all_masks = np.stack(masks + random_masks)
    prob, _ = predictor.predict_masks(sample, all_masks)
    target = modal["target"]
    changes = float(modal["values_cls"][-1]) - score_class(prob, target)
    observed = float(np.mean(changes[:len(masks)]))
    random_mean = float(np.mean(changes[len(masks):])) if random_masks else None
    return dict(aopc=observed, random_aopc=random_mean,
                delta_aopc=observed - random_mean if random_mean is not None else None,
                steps=len(masks))


def validation(args, predictor, model: str, config_hash: str) -> None:
    # Q2 field-major loader and train-fitted per-seed scaler remain the sole data path.
    from src.q2.data import load_aligned
    valid = load_aligned(args.data)["valid"]
    output = Path(args.output)
    rows, details, cards, importance_rows, fidelity_rows = [], [], [], [], []
    probabilities, intensities = [], []
    for i, sample_id in enumerate(valid["id"]):
        sample = slice_sample(valid, i)
        modal, evidence, importance = explain_one(predictor, sample, args.window)
        rows.append(main_row(str(sample_id), model, modal, config_hash, "not_applicable"))
        probabilities.append(modal["probabilities"])
        intensities.append(modal["intensity"])
        for item in evidence:
            details.append(dict(sample_id=str(sample_id), model_id=model, **item))
        for m, name in enumerate(MODALITIES):
            for t in np.flatnonzero(sample["raw_available"][0, m]):
                importance_rows.append(dict(sample_id=str(sample_id), modality=name,
                                            position=int(t), importance=float(importance[m, t])))
        if args.fidelity:
            fidelity_rows.append(dict(sample_id=str(sample_id), **fidelity(
                predictor, sample, modal, evidence, args.random_repeats, args.seed + i)))
        cards.append(dict(sample_id=str(sample_id), true_cls=int(valid["label_cls"][i]),
                          true_reg=float(valid["label_reg"][i]), prediction=rows[-1],
                          evidence=evidence))
        if (i + 1) % 25 == 0:
            print(f"valid {i + 1}/{len(valid['id'])}", flush=True)
    probabilities = np.asarray(probabilities)
    intensities = np.asarray(intensities)
    report = scores(valid["label_cls"], probabilities.argmax(1), valid["label_reg"], intensities)
    report.update(model_id=model, explanation_count=len(rows))
    if args.expect_q2:
        expected = dict(accuracy=.6428571428571429, macro_f1=.6279174968631847,
                        mae=.5659820862193745, pearson=.6821057476686703)
        mismatches = {key: (report[key], value) for key, value in expected.items()
                      if report[key] is None or abs(report[key] - value) > args.metric_tolerance}
        if mismatches:
            raise ValueError(f"Q2 clean reproduction failed: {mismatches}")
    write_csv(output / "valid_predictions_explanations.csv", rows, MAIN_COLUMNS)
    write_csv(output / "valid_evidence_detail.csv", details,
              tuple(dict.fromkeys(EVIDENCE_COLUMNS + ("modality_index",))))
    write_csv(output / "valid_position_importance.csv", importance_rows,
              ("sample_id", "modality", "position", "importance"))
    if args.fidelity:
        write_csv(output / "valid_fidelity.csv", fidelity_rows,
                  ("sample_id", "aopc", "random_aopc", "delta_aopc", "steps"))
        differences = np.array([r["delta_aopc"] for r in fidelity_rows
                                if r["delta_aopc"] is not None], dtype=float)
        if len(differences):
            rng = np.random.default_rng(args.seed)
            boot = np.mean(rng.choice(differences, size=(2000, len(differences)), replace=True), axis=1)
            report["fidelity"] = dict(n=len(differences), mean_delta_aopc=float(np.mean(differences)),
                                      ci95=np.quantile(boot, [.025, .975]).tolist())
    (output / "valid_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "valid_explanation_cards.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("attachment4", "validate"))
    parser.add_argument("--checkpoints", nargs=3, type=Path, required=True,
                        metavar=("SEED3407", "SEED42", "SEED2026"))
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="attachment-4 aligned-version directory")
    parser.add_argument("--data", type=Path, help="attachment-2 aligned_50.pkl")
    parser.add_argument("--q2-root", type=Path)
    parser.add_argument("--q1-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--align-device", default="cpu")
    parser.add_argument("--align", action="store_true", help="run Q1 frozen forced alignment on attachment 4")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--fidelity", action="store_true")
    parser.add_argument("--expect-q2", action="store_true",
                        help="require the current weighted-M4 ensemble valid metrics")
    parser.add_argument("--metric-tolerance", type=float, default=1e-4)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.window < 1 or args.random_repeats < 1:
        parser.error("window and random-repeats must be positive")
    if args.mode == "attachment4" and not args.input:
        parser.error("attachment4 requires --input")
    if args.mode == "validate" and not args.data:
        parser.error("validate requires --data")
    config = dict(mode=args.mode, checkpoints=[str(x) for x in args.checkpoints],
                  bert=str(args.bert), window=args.window, align=args.align,
                  fidelity=args.fidelity, random_repeats=args.random_repeats, seed=args.seed,
                  expect_q2=args.expect_q2, metric_tolerance=args.metric_tolerance)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    predictor = Ensemble.from_checkpoints(args.checkpoints, args.bert, args.device, args.q2_root)
    identifier = model_id(args.checkpoints, args.bert)
    if args.mode == "attachment4":
        attachment4(args, predictor, identifier, config_hash)
    else:
        validation(args, predictor, identifier, config_hash)


if __name__ == "__main__":
    main()
