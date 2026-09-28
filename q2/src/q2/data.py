"""Field-major Pickle adapter and data gates for the aligned_50 track."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

FIELDS = ("id", "text_bert", "audio", "vision", "classification_labels", "regression_labels")
SHAPES = {"text_bert": (3, 50), "audio": (50, 74), "vision": (50, 35)}


def content_mask(bert: np.ndarray) -> np.ndarray:
    ids, attention = bert[:, 0, :], bert[:, 1, :]
    return (attention != 0) & (ids != 101) & (ids != 102)


def prepare_split(source: dict, split: str) -> dict:
    """Read one sample as source[split][field][j], never source[split][j][field]."""
    s = source[split]
    if not isinstance(s, dict) or not set(FIELDS).issubset(s):
        raise ValueError(f"{split}: expected field-major dictionary with {FIELDS}")
    n = len(s["id"])
    if any(len(s[k]) != n for k in FIELDS):
        raise ValueError(f"{split}: field lengths disagree")
    arrays = {k: np.asarray(s[k]) for k in FIELDS if k != "id"}
    for k, expected in SHAPES.items():
        if arrays[k].shape != (n, *expected):
            raise ValueError(f"{split}.{k}: {arrays[k].shape} != {(n, *expected)}")
    bert = arrays["text_bert"].astype(np.int64, copy=False)
    ids, attention, types = bert[:, 0], bert[:, 1], bert[:, 2]
    if not np.isin(attention, [0, 1]).all() or not np.isin(types, [0, 1]).all():
        raise ValueError(f"{split}: invalid BERT mask or token types")
    if (ids[:, 0] != 101).any() or ((attention.sum(1) < 2)).any():
        raise ValueError(f"{split}: invalid CLS/SEP structure")
    if any(ids[i, int(attention[i].sum()) - 1] != 102 for i in range(n)):
        raise ValueError(f"{split}: SEP is not at last attended position")
    content = content_mask(bert)
    audio = arrays["audio"].astype(np.float32, copy=False)
    vision = arrays["vision"].astype(np.float32, copy=False)
    raw = np.stack((content, content & np.any(audio != 0, 2), content & np.any(vision != 0, 2)), axis=1)
    cls_float = arrays["classification_labels"].reshape(n)
    cls = cls_float.astype(np.int64)
    reg = arrays["regression_labels"].reshape(n).astype(np.float32)
    if not np.array_equal(cls_float, cls) or not np.isin(cls, [0, 1, 2]).all():
        raise ValueError(f"{split}: invalid class labels")
    if not np.isfinite(reg).all() or (np.abs(reg) > 3.0001).any():
        raise ValueError(f"{split}: invalid regression labels")
    if not np.isfinite(audio).all() or not np.isfinite(vision).all():
        raise ValueError(f"{split}: nonfinite A/V values")
    return dict(id=np.asarray(s["id"], dtype=str), text_bert=bert,
                audio=audio, vision=vision, content=content, raw_available=raw,
                label_cls=cls, label_reg=reg)


def load_aligned(path: str | Path) -> dict:
    with open(path, "rb") as f:
        source = pickle.load(f)  # Only trusted competition files.
    if set(source) != {"train", "valid", "test"}:
        raise ValueError("aligned pickle must have train/valid/test keys")
    return {name: prepare_split(source, name) for name in ("train", "valid", "test")}


class RobustAVScaler:
    def __init__(self):
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, train: dict):
        for j, name in ((1, "audio"), (2, "vision")):
            observed = train[name][train["raw_available"][:, j]]
            median = np.median(observed, axis=0)
            q25, q75 = np.percentile(observed, [25, 75], axis=0)
            scale = np.maximum((q75 - q25) / 1.349, 1e-6)
            self.stats[name] = median.astype(np.float32), scale.astype(np.float32)
        return self

    def transform(self, split: dict) -> dict:
        out = dict(split)
        for j, name in ((1, "audio"), (2, "vision")):
            median, scale = self.stats[name]
            x = np.clip((split[name] - median) / scale, -5, 5)
            out[name] = np.where(split["raw_available"][:, j, :, None], x, 0).astype(np.float32)
        return out

    def save(self, path: str | Path):
        np.savez(path, **{f"{name}_{part}": v for name, pair in self.stats.items()
                          for part, v in zip(("median", "scale"), pair)})

    @classmethod
    def load(cls, path: str | Path):
        obj = cls()
        with np.load(path) as z:
            for name in ("audio", "vision"):
                obj.stats[name] = z[f"{name}_median"], z[f"{name}_scale"]
        return obj


def audit(path: str | Path, output: str | Path):
    data = load_aligned(path)
    ids = [str(x) for s in data.values() for x in s["id"]]
    if len(ids) != len(set(ids)):
        raise ValueError("sample ID overlap or duplicate")
    report = {}
    for name, s in data.items():
        report[name] = dict(count=len(s["id"]), classes=np.bincount(s["label_cls"], minlength=3).tolist(),
                            content_positions=int(s["content"].sum()),
                            audio_observed=int(s["raw_available"][:, 1].sum()),
                            vision_observed=int(s["raw_available"][:, 2].sum()),
                            max_token_id=int(s["text_bert"][:, 0].max()),
                            label_min=float(s["label_reg"].min()), label_max=float(s["label_reg"].max()))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf8")
    return data, report
