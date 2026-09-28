"""Predeclared Q2 duration, morphology, synchronization and stress probes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .data import load_aligned
from .missing import MODALITIES, SUBSETS, make_mask
from .run import evaluate, load_frozen


def fragmented(content, subset, blocks, ratio=.25):
    idx = np.flatnonzero(content)
    mask = np.zeros((3, len(content)), bool)
    count = min(max(1, round(ratio * len(idx))), max(0, len(idx) - 1))
    if count < blocks or len(idx) - count < blocks - 1:
        return None
    sizes = np.full(blocks, count // blocks, int)
    sizes[:count % blocks] += 1
    spare = len(idx) - count - (blocks - 1)
    before = spare // 2
    selected = []
    cursor = before
    for size in sizes:
        selected.extend(idx[cursor:cursor + size])
        cursor += size + 1
    for m in subset:
        mask[m, selected] = True
    return mask


def make_conditions(content_batch, seed=3407):
    rng = np.random.default_rng(seed)
    length = content_batch.shape[-1]
    conditions = {}
    for subset in SUBSETS:
        name = "+".join(MODALITIES[m] for m in subset)
        fragment_arrays = {blocks: [fragmented(c, subset, blocks) for c in content_batch]
                           for blocks in (1, 2, 4)}
        # Same samples across 1/2/4 blocks: otherwise morphology is confounded
        # by short-sequence exclusion.
        common_eligible = np.array([all(fragment_arrays[b][i] is not None for b in (1, 2, 4))
                                    for i in range(len(content_batch))])
        for blocks, arrays in fragment_arrays.items():
            masks = np.stack([x if x is not None else np.zeros((3, length), bool) for x in arrays])
            conditions[f"fragment_{name}_{blocks}"] = (masks, common_eligible)
        if len(subset) == 1:
            for gap in (1, 2, 3, 5, 8):
                arrays, eligible = [], []
                for content in content_batch:
                    idx = np.flatnonzero(content)
                    valid = gap < len(idx)
                    eligible.append(valid)
                    mask = np.zeros((3, length), bool)
                    if valid:
                        first = (len(idx) - gap) // 2
                        mask[subset[0], idx[first:first + gap]] = True
                    arrays.append(mask)
                conditions[f"gap_{name}_{gap}"] = (np.stack(arrays), np.array(eligible))
            iid, block = [], []
            for content in content_batch:
                idx = np.flatnonzero(content)
                count = min(len(idx) - 1, max(1, round(.25 * len(idx))))
                mask = np.zeros((3, length), bool)
                if count > 0:
                    mask[subset[0], rng.choice(idx, count, replace=False)] = True
                iid.append(mask)
                block.append(make_mask(content, subset, .25, "middle"))
            conditions[f"iid_{name}"] = (np.stack(iid), np.ones(len(iid), bool))
            conditions[f"block_{name}"] = (np.stack(block), np.ones(len(block), bool))
        else:
            sync, asynchronous = [], []
            for content in content_batch:
                first = make_mask(content, (subset[0],), .25, "random", rng=rng)
                second = make_mask(content, (subset[1],), .25, "random", rng=rng)
                same = np.zeros_like(first)
                same[subset[0]] = first[subset[0]]
                same[subset[1]] = first[subset[0]]
                separate = first.copy()
                separate[subset[1]] = second[subset[1]]
                sync.append(same)
                asynchronous.append(separate)
            conditions[f"sync_{name}"] = (np.stack(sync), np.ones(len(sync), bool))
            conditions[f"async_{name}"] = (np.stack(asynchronous), np.ones(len(asynchronous), bool))
    for ratio in (.10, .25):
        masks = np.stack([make_mask(c, (0, 1, 2), ratio, "middle") for c in content_batch])
        conditions[f"triple_{int(ratio * 100)}"] = (masks, np.ones(len(masks), bool))
    return conditions


def max_gap(mask):
    longest = 0
    for row in mask:
        running = 0
        for value in row:
            running = running + 1 if value else 0
            longest = max(longest, running)
    return longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bert")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler = load_frozen(Path(args.checkpoint), device, args.bert)
    valid = scaler.transform(load_aligned(args.data)["valid"])
    conditions = make_conditions(valid["content"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_id", "sample_id", "true_cls", "true_reg", "pred_cls", "pred_reg",
                         "prob_negative", "prob_neutral", "prob_positive"])
        for name, (mask, eligible) in conditions.items():
            idx = np.flatnonzero(eligible)
            subset = {k: v[idx] for k, v in valid.items()}
            metrics, probs, scores = evaluate(model, subset, mask[idx], device)
            rows.append(dict(condition_id=name, n=len(idx), max_gap=float(np.mean([max_gap(m) for m in mask[idx]])), **metrics))
            for j, sample_id in enumerate(subset["id"]):
                writer.writerow([name, sample_id, int(subset["label_cls"][j]), float(subset["label_reg"][j]),
                                 int(probs[j].argmax()), float(scores[j]), *map(float, probs[j])])
            print(name, metrics, flush=True)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
