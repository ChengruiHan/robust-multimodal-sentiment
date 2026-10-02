"""Synthetic metric examples and evidence validation failures; no research data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release_evidence import score, verify_inputs


def row(name, y, yp, z, zp):
    return dict(sample_id=name, true_cls=y, pred_cls=yp, true_reg=z, pred_reg=zp,
                **dict(zip(("prob_negative", "prob_neutral", "prob_positive"),
                           [0.8 if k == yp else 0.1 for k in range(3)])))


class ReleaseEvidenceTest(unittest.TestCase):
    def test_known_confusion_and_regression_errors(self):
        result = score([row("a", 0, 0, -1, -1), row("b", 1, 2, 0, 0.3),
                        row("c", 2, 2, 1, 0.8)])
        self.assertEqual(result["confusion"], [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["macro_f1"], 5/9)
        self.assertAlmostEqual(result["mae"], 1/6)

    def test_duplicate_and_inconsistent_prediction_rejected(self):
        r = row("a", 0, 0, -1, -1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score([r, r])
        r["pred_cls"] = 2
        with self.assertRaisesRegex(ValueError, "argmax"):
            score([r])

    def test_nonfinite_prediction_rejected(self):
        r = row("a", 0, 0, -1, float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            score([r])

    def test_altered_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "record.csv").write_text("changed")
            (root / "INPUTS.sha256.json").write_text(json.dumps({"files": {"record.csv": "0"*64}}))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_inputs(root)



if __name__ == "__main__":
    unittest.main()
