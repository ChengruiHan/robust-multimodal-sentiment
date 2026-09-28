"""Cache frozen M4 representations for the final conditional regression head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import load_aligned
from .missing import grid55
from .run import batches, load_frozen, to_torch


MINI = ("T_10_front", "T_25_middle", "T_40_end", "A_25_middle",
        "V_25_middle", "A+V_25_middle")


def pooled_features(model, split, drop, device, batch_size=64):
    parts, scores = [], []
    model.eval()
    with torch.inference_mode():
        for idx, batch in batches(split, batch_size):
            out = model(**to_torch(batch, drop[idx], device))
            parts.append(out["pooled"].cpu())
            scores.append(out["pred_reg"].clamp(-3, 3).cpu().numpy())
    return torch.cat(parts), np.concatenate(scores)


def create_cache(base: Path, data_path: Path, bert: Path, path: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler = load_frozen(base, device, str(bert))
    if model.variant != "M4" or model.decision_head != "flat" or model.bert_train_last_n:
        raise ValueError("requires frozen-BERT, flat-head M4")
    data = load_aligned(data_path)
    train = scaler.transform(data["train"])
    valid = scaler.transform(data["valid"])
    train_drop = np.zeros_like(train["raw_available"])
    train_pooled, _ = pooled_features(model, train, train_drop, device)
    masks = grid55(valid["content"])
    valid_pooled = {}
    for name in ("clean", *MINI):
        valid_pooled[name], _ = pooled_features(model, valid, masks[name], device)
    cache = dict(base_checkpoint=str(base.resolve()),
                 train_pooled=train_pooled, train_reg=torch.as_tensor(train["label_reg"]),
                 valid_pooled=valid_pooled,
                 valid_reg=torch.as_tensor(valid["label_reg"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"cached {len(train_pooled)} train and {len(valid['id'])} valid samples: {path}",
          flush=True)
    return cache
