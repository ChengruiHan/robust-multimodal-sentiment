"""Verify fixed task checkpoints and the frozen BERT file before inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(root: Path, bert_dir: Path, weights_dir: Path | None = None) -> None:
    manifest = json.loads((root / "config/final_model.json").read_text(encoding="utf-8"))
    weights_dir = weights_dir if weights_dir is not None else root / "weights"
    for filename, expected in manifest["weights_sha256"].items():
        paths = ([weights_dir / filename] if filename != "scaler.npz"
                 else [weights_dir / f"seed_{seed}" / filename
                       for seed in manifest["seeds"]])
        for path in paths:
            if sha256(path) != expected:
                raise ValueError(f"SHA-256 mismatch: {path}")
    model = bert_dir / "model.safetensors"
    if sha256(model) != manifest["bert_model_safetensors_sha256"]:
        raise ValueError(f"SHA-256 mismatch: {model}")
    if not (bert_dir / "config.json").is_file():
        raise ValueError("BERT config.json missing")
    print("verified three Q2 checkpoints, scalers and frozen BERT model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weights-dir", type=Path,
                        help="Frozen weights directory; defaults to ROOT/weights")
    args = parser.parse_args()
    check(args.root, args.bert_dir, args.weights_dir)
