"""TRACE-MSA interventions and exact three-player attribution.

The frozen STATE-MSA predictor is loaded only when Ensemble.from_checkpoints is
called, so the attribution arithmetic can be tested without torch/BERT.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

MODALITIES = ("text", "audio", "vision")
COALITIONS = tuple(tuple(c) for n in range(4) for c in combinations(range(3), n))


def require_sample(sample: dict) -> None:
    for name, shape in (("text_bert", (1, 3, 50)), ("audio", (1, 50, 74)),
                        ("vision", (1, 50, 35)), ("content", (1, 50)),
                        ("raw_available", (1, 3, 50))):
        if np.asarray(sample[name]).shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {np.asarray(sample[name]).shape}")
    if not np.array_equal(sample["raw_available"][:, 0], sample["content"]):
        raise ValueError("text availability must equal valid content positions")


def slice_sample(split: dict, index: int) -> dict:
    return {key: np.asarray(value)[index:index + 1] for key, value in split.items()
            if key in ("id", "text_bert", "audio", "vision", "content", "raw_available",
                       "label_cls", "label_reg")}


def masks_for_coalitions(sample: dict) -> np.ndarray:
    """One mask per coalition; absent modalities lose every observed content position."""
    raw = sample["raw_available"][0]
    masks = np.zeros((8, 3, 50), dtype=bool)
    for row, coalition in enumerate(COALITIONS):
        for modality in range(3):
            if modality not in coalition:
                masks[row, modality] = raw[modality]
    return masks


def score_class(probabilities: np.ndarray, target_class: int) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64)[..., target_class], 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def shapley(values: np.ndarray) -> np.ndarray:
    """Exact Shapley values for coalition order COALITIONS; arbitrary leading axes."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 8:
        raise ValueError("expected eight coalition values")
    where = {coalition: n for n, coalition in enumerate(COALITIONS)}
    phi = np.zeros(values.shape[:-1] + (3,), dtype=np.float64)
    weights = (1 / 3, 1 / 6, 1 / 3)
    for m in range(3):
        others = [j for j in range(3) if j != m]
        for size in range(3):
            for subset in combinations(others, size):
                with_m = tuple(sorted(subset + (m,)))
                phi[..., m] += weights[size] * (values[..., where[with_m]] - values[..., where[subset]])
    if not np.allclose(phi.sum(-1), values[..., -1] - values[..., 0], atol=1e-8):
        raise AssertionError("Shapley efficiency failed")
    return phi


def contribution_summary(phi: np.ndarray, tolerance: float = 1e-8) -> dict:
    phi = np.asarray(phi, dtype=np.float64)
    total = float(np.abs(phi).sum())
    if total <= tolerance:
        return dict(shares=[None] * 3, primary="undetermined", primary_sign="none",
                    main_support="undetermined", weak=True)
    shares = (np.abs(phi) / total).tolist()
    primary = int(np.abs(phi).argmax())
    positive = np.flatnonzero(phi > tolerance)
    support = MODALITIES[int(positive[np.argmax(phi[positive])])] if len(positive) else "undetermined"
    return dict(shares=shares, primary=MODALITIES[primary],
                primary_sign="support" if phi[primary] > tolerance else "oppose",
                main_support=support, weak=False)


@dataclass
class Ensemble:
    models: list
    scalers: list
    device: str
    q2_to_torch: object

    @classmethod
    def from_checkpoints(cls, checkpoints: list[str | Path], bert: str | Path,
                         device: str = "cpu", q2_root: str | Path | None = None) -> "Ensemble":
        if len(checkpoints) != 3:
            raise ValueError("Q3 requires exactly the three frozen Q2 checkpoints")
        root = Path(q2_root) if q2_root else Path(__file__).resolve().parents[2] / "vendor" / "q2_code"
        if not (root / "src" / "q2" / "run.py").is_file():
            raise FileNotFoundError(f"Q2 code not found: {root}")
        sys.path.insert(0, str(root))
        import src
        if str(root / "src") not in src.__path__:
            src.__path__.append(str(root / "src"))
        from src.q2.run import load_frozen, to_torch
        models, scalers = [], []
        for path in checkpoints:
            model, scaler = load_frozen(Path(path), device, str(bert))
            if model.variant != "M4" or model.reg_head_type != "conditional":
                raise ValueError(f"expected M4: {path}")
            models.append(model)
            scalers.append(scaler)
        return cls(models, scalers, device, to_torch)

    def predict_masks(self, sample: dict, masks: np.ndarray, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Return probabilities and Q2-clipped intensity for every intervention."""
        require_sample(sample)
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != (3, 50):
            raise ValueError("masks must have shape [N,3,50]")
        import torch
        n = len(masks)
        seed_prob, seed_reg = [], []
        for model, scaler in zip(self.models, self.scalers):
            scaled = scaler.transform(sample)
            probs, regs = [], []
            model.eval()
            with torch.inference_mode():
                for start in range(0, n, batch_size):
                    end = min(n, start + batch_size)
                    batch = {key: np.repeat(scaled[key], end - start, axis=0)
                             for key in ("text_bert", "audio", "vision", "content", "raw_available")}
                    feed = self.q2_to_torch(batch, masks[start:end], self.device)
                    out = model(**feed)
                    probs.append(out["logits"].softmax(-1).cpu().numpy())
                    regs.append(out["pred_reg"].clamp(-3, 3).cpu().numpy())
            seed_prob.append(np.concatenate(probs))
            seed_reg.append(np.concatenate(regs))
        return np.mean(seed_prob, axis=0), np.mean(seed_reg, axis=0)


def explain_modalities(predictor, sample: dict) -> dict:
    masks = masks_for_coalitions(sample)
    probabilities, intensity = predictor.predict_masks(sample, masks)
    full = COALITIONS.index((0, 1, 2))
    target = int(probabilities[full].argmax())
    class_values = score_class(probabilities, target)
    phi_cls, phi_reg = shapley(class_values), shapley(intensity)
    absent = [m for m in range(3) if not sample["raw_available"][0, m].any()]
    for m in absent:
        # A modality with no observed positions is a dummy player. Floating
        # point inference may leave tiny nonzero values across equal coalitions.
        phi_cls[m] = 0.0
        phi_reg[m] = 0.0
    return dict(probabilities=probabilities[full], intensity=float(intensity[full]),
                target=target, phi_cls=phi_cls, phi_reg=phi_reg,
                values_cls=class_values, values_reg=intensity,
                summary=contribution_summary(phi_cls),
                raw_unavailable=[MODALITIES[m] for m in absent])


def local_windows(sample: dict, width: int = 3) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Use only consecutive observed positions; half-open 0-based intervals."""
    raw = sample["raw_available"][0]
    windows = []
    for m in range(3):
        indices = np.flatnonzero(raw[m])
        if not len(indices):
            continue
        actual_width = min(width, len(indices))
        for start in range(0, 50 - actual_width + 1):
            interval = np.arange(start, start + actual_width)
            if np.all(raw[m, interval]):
                windows.append((m, start, start + actual_width))
    masks = np.zeros((len(windows), 3, 50), dtype=bool)
    for i, (m, start, end) in enumerate(windows):
        masks[i, m, start:end] = True
    return masks, windows


def explain_windows(predictor, sample: dict, modal: dict, width: int = 3) -> list[dict]:
    masks, windows = local_windows(sample, width)
    if not len(masks):
        return []
    prob, reg = predictor.predict_masks(sample, masks)
    base_cls = float(modal["values_cls"][-1])
    base_reg = float(modal["intensity"])
    scores = score_class(prob, modal["target"])
    return [dict(modality=MODALITIES[m], modality_index=m, start_idx=start, end_idx=end,
                 delta_cls=base_cls - float(scores[i]), delta_reg=base_reg - float(reg[i]))
            for i, (m, start, end) in enumerate(windows)]


def select_evidence(windows: list[dict], limit_per_sign: int = 2, overlap_threshold: float = .5,
                    tolerance: float = 1e-8) -> list[dict]:
    selected = []
    for modality in MODALITIES:
        for sign, direction in (("support", 1), ("oppose", -1)):
            options = [w for w in windows if w["modality"] == modality
                       and direction * w["delta_cls"] > tolerance]
            options.sort(key=lambda w: (-abs(w["delta_cls"]), w["start_idx"], w["end_idx"]))
            keep = []
            for candidate in options:
                a = set(range(candidate["start_idx"], candidate["end_idx"]))
                if any(len(a & set(range(w["start_idx"], w["end_idx"]))) / len(a | set(range(w["start_idx"], w["end_idx"])))
                       > overlap_threshold for w in keep):
                    continue
                keep.append(candidate)
                if len(keep) == limit_per_sign:
                    break
            for rank, item in enumerate(keep, 1):
                selected.append({**item, "evidence_type": sign, "rank": rank})
    return selected


def position_importance(windows: list[dict]) -> np.ndarray:
    total = np.zeros((3, 50), dtype=float)
    count = np.zeros((3, 50), dtype=int)
    for w in windows:
        m, start, end = w["modality_index"], w["start_idx"], w["end_idx"]
        total[m, start:end] += w["delta_cls"]
        count[m, start:end] += 1
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
