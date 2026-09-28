"""Pure numerical QTOP implementation with auditable source-frame CSR maps."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FrameSeries:
    values: np.ndarray
    centers: np.ndarray
    quality: np.ndarray
    source_ids: np.ndarray
    span_start: float
    span_end: float


def support_cells(centers: np.ndarray, start: float, end: float) -> np.ndarray:
    """Nonoverlapping midpoint cells, clipped to the measured stream span."""
    t = np.asarray(centers, dtype=np.float64)
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("invalid stream span")
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("frame centers must be finite and strictly increasing")
    if len(t) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if t[0] < start - 1e-6 or t[-1] > end + 1e-6:
        raise ValueError("frame center outside stream span")
    boundaries = np.r_[start, (t[:-1] + t[1:]) / 2, end]
    return np.column_stack((boundaries[:-1], boundaries[1:]))


def aggregate(starts: np.ndarray, ends: np.ndarray, valid: np.ndarray,
              series: FrameSeries, feature_dim: int) -> dict[str, np.ndarray]:
    """Aggregate each valid word interval by overlap times frame quality.

    Frame maps include only frames with positive overlap and quality. Zero quality
    frames still enter the denominator of mean observed quality.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    values = np.asarray(series.values, dtype=np.float64)
    centers = np.asarray(series.centers, dtype=np.float64)
    quality = np.asarray(series.quality, dtype=np.float64)
    ids = np.asarray(series.source_ids, dtype=np.int64)
    length = len(starts)
    if any(len(a) != length for a in (ends, valid)):
        raise ValueError("word arrays have different lengths")
    if values.shape != (len(centers), feature_dim) or len(quality) != len(centers) or len(ids) != len(centers):
        raise ValueError("frame series shapes disagree")
    if not np.isfinite(values).all() or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
        raise ValueError("invalid frame values or quality")
    if np.any(valid & (~np.isfinite(starts) | ~np.isfinite(ends) | (starts >= ends))):
        raise ValueError("invalid word intervals")
    cells = support_cells(centers, series.span_start, series.span_end)
    out = np.zeros((length, 2 * feature_dim), dtype=np.float32)
    available = np.zeros(length, dtype=bool)
    coverage = np.zeros(length, dtype=np.float32)
    mean_quality = np.zeros(length, dtype=np.float32)
    dispersion_valid = np.zeros(length, dtype=bool)
    offsets = [0]
    frame_ids: list[int] = []
    for k in range(length):
        if valid[k] and len(cells):
            overlap = np.maximum(0, np.minimum(ends[k], cells[:, 1]) - np.maximum(starts[k], cells[:, 0]))
            weights = overlap * quality
            chosen = np.flatnonzero(weights > 0)
            frame_ids.extend(ids[chosen].tolist())
            covered = overlap[quality > 0].sum()
            coverage[k] = min(1.0, covered / (ends[k] - starts[k]))
            if overlap.sum() > 0:
                mean_quality[k] = np.dot(overlap, quality) / overlap.sum()
            if len(chosen):
                available[k] = True
                alpha = weights[chosen] / weights[chosen].sum()
                mu = np.sum(alpha[:, None] * values[chosen], axis=0)
                var = np.sum(alpha[:, None] * (values[chosen] - mu) ** 2, axis=0)
                out[k] = np.r_[mu, np.sqrt(np.maximum(var, 0))]
                dispersion_valid[k] = len(chosen) > 1
        offsets.append(len(frame_ids))
    return {
        "features": out,
        "available": available,
        "coverage": coverage,
        "quality": mean_quality,
        "dispersion_valid": dispersion_valid,
        "frame_offsets": np.asarray(offsets, dtype=np.int64),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "support_cells": cells,
    }
