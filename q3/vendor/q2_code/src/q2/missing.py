"""Deterministic content-only synthetic gaps."""
from __future__ import annotations

import numpy as np

MODALITIES = ("T", "A", "V")
SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))


def _one_block(indices: np.ndarray, count: int, position: str, rng: np.random.Generator) -> np.ndarray:
    if position == "front":
        start = 0
    elif position == "end":
        start = len(indices) - count
    elif position == "middle":
        start = (len(indices) - count) // 2
    elif position == "random":
        start = int(rng.integers(0, len(indices) - count + 1))
    else:
        raise ValueError(position)
    return indices[start:start + count]


def make_mask(content: np.ndarray, subset: tuple[int, ...], ratio: float,
              position: str = "middle", blocks: int = 1, sync: bool = True,
              rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    indices = np.flatnonzero(content)
    drop = np.zeros((3, len(content)), dtype=bool)
    if len(indices) <= 1 or not subset:
        return drop
    count = min(len(indices) - 1, max(1, round(ratio * len(indices))))

    def draw() -> np.ndarray:
        if blocks == 1 or count < 2 or len(indices) - count < 1:
            return _one_block(indices, count, position, rng)
        first = max(1, count // 2)
        second = count - first
        # A positive gap separates two blocks. Placement remains deterministic.
        max_left = len(indices) - count - 1
        if max_left < 0:
            return _one_block(indices, count, position, rng)
        if position == "front":
            left = 0
        elif position == "end":
            left = max_left
        elif position == "middle":
            left = max_left // 2
        elif position == "random":
            left = int(rng.integers(0, max_left + 1))
        else:
            raise ValueError(position)
        middle_gap = 1
        return np.r_[indices[left:left + first],
                     indices[left + first + middle_gap:left + first + middle_gap + second]]

    shared = draw() if sync else None
    for m in subset:
        drop[m, shared if sync else draw()] = True
    return drop


def mixed_masks(content_batch: np.ndarray, rng: np.random.Generator,
                corrupt_probability: float = .75) -> np.ndarray:
    out = np.zeros((len(content_batch), 3, content_batch.shape[1]), dtype=bool)
    for i, content in enumerate(content_batch):
        if rng.random() >= corrupt_probability:
            continue
        subset = SUBSETS[int(rng.integers(len(SUBSETS)))]
        out[i] = make_mask(content, subset, float(rng.uniform(.05, .50)),
                           ("front", "middle", "end", "random")[int(rng.integers(4))],
                           1 if rng.random() < .7 else 2, bool(rng.integers(2)), rng)
    return out


def grid55(content_batch: np.ndarray) -> dict[str, np.ndarray]:
    result = {"clean": np.zeros((len(content_batch), 3, content_batch.shape[1]), dtype=bool)}
    for subset in SUBSETS:
        name = "+".join(MODALITIES[x] for x in subset)
        for ratio in (.10, .25, .40):
            for position in ("front", "middle", "end"):
                key = f"{name}_{int(ratio * 100)}_{position}"
                result[key] = np.stack([make_mask(c, subset, ratio, position) for c in content_batch])
    assert len(result) == 55
    return result
