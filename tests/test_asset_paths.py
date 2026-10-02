"""Frozen-asset verification uses the same directory selected for inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_check_assets", ROOT / "sentiment_model/scripts/check_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetPathsTest(unittest.TestCase):
    def test_external_weights_and_all_three_scalers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "code"
            external = Path(folder) / "external"
            bert = Path(folder) / "bert"
            (root / "config").mkdir(parents=True)
            bert.mkdir()
            content = b"fixture"
            digest = hashlib.sha256(content).hexdigest()
            seeds = [3407, 42, 2026]
            for seed in seeds:
                path = external / f"seed_{seed}"
                path.mkdir(parents=True)
                (path / "best.pt").write_bytes(content)
                (path / "scaler.npz").write_bytes(content)
            (bert / "model.safetensors").write_bytes(content)
            (bert / "config.json").write_text("{}")
            manifest = dict(seeds=seeds, weights_sha256={
                **{f"seed_{s}/best.pt": digest for s in seeds}, "scaler.npz": digest},
                bert_model_safetensors_sha256=digest)
            (root / "config/final_model.json").write_text(json.dumps(manifest))
            assets.check(root, bert, external)
            (external / "seed_42/scaler.npz").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                assets.check(root, bert, external)
            with self.assertRaises(FileNotFoundError):
                assets.check(root, bert)


if __name__ == "__main__":
    unittest.main()
