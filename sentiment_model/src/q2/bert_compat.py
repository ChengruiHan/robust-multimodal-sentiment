"""P0 gate: verify local bert-base-uncased reproduces attachment-2 text."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    with open(args.data, "rb") as f:
        split = pickle.load(f)["train"]
    rng = np.random.default_rng(3407)
    indices = np.sort(rng.choice(len(split["id"]), min(args.samples, len(split["id"])), replace=False))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertModel.from_pretrained(args.bert, local_files_only=True).to(device).eval()
    errors, cosine = [], []
    mismatches = []
    with torch.inference_mode():
        for start in range(0, len(indices), 16):
            idx = indices[start:start + 16]
            bert = np.asarray(split["text_bert"][idx], dtype=np.int64)
            actual = np.asarray(split["text"][idx], dtype=np.float32)
            output = model(input_ids=torch.as_tensor(bert[:, 0], device=device),
                           attention_mask=torch.as_tensor(bert[:, 1], device=device),
                           token_type_ids=torch.as_tensor(bert[:, 2], device=device)).last_hidden_state.cpu().numpy()
            difference = np.abs(output - actual)
            sample_mae = difference.mean((1, 2))
            sample_max = difference.max((1, 2))
            numerator = (output * actual).sum((1, 2))
            denominator = np.linalg.norm(output.reshape(len(idx), -1), axis=1) * np.linalg.norm(actual.reshape(len(idx), -1), axis=1)
            sample_cos = numerator / np.maximum(denominator, 1e-9)
            errors.extend(zip(sample_mae.tolist(), sample_max.tolist()))
            cosine.extend(sample_cos.tolist())
            for j, mae in enumerate(sample_mae):
                if mae > 1e-3:
                    mismatches.append(dict(sample_id=str(split["id"][idx[j]]), mae=float(mae), max_abs=float(sample_max[j])))
    metrics = dict(samples=len(indices), mean_abs=float(np.mean([x[0] for x in errors])),
                   max_abs=float(max(x[1] for x in errors)), median_cosine=float(np.median(cosine)),
                   mismatched_over_1e_3=len(mismatches), top_mismatches=sorted(mismatches, key=lambda x:-x["mae"])[:10])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
