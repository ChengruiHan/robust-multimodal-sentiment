import tempfile
import unittest
from pathlib import Path

from scripts.q1_policy import alignment_gate, load_review_files


class ReviewGateTest(unittest.TestCase):
    def test_silence_precedes_review_and_review_needs_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / "candidates.csv"
            review = Path(folder) / "review.csv"
            candidate.write_text("sample_id,reason\na$_$1,reported\n", encoding="utf-8")
            review.write_text("sample_id,decision,note,evidence_time_s\n", encoding="utf-8")
            candidates, decisions = load_review_files(candidate, review, {"a$_$1", "b$_$2"})
            self.assertEqual(alignment_gate(0, "a$_$1", candidates, decisions), "SKIP_SILENT")
            self.assertEqual(alignment_gate(20, "a$_$1", candidates, decisions), "REVIEW_REQUIRED")
            self.assertEqual(alignment_gate(20, "b$_$2", candidates, decisions), "AUTO_UNVERIFIED")
            review.write_text("sample_id,decision,note,evidence_time_s\na$_$1,BLOCK_ALIGN,heard other words,2.1\n")
            candidates, decisions = load_review_files(candidate, review, {"a$_$1", "b$_$2"})
            self.assertEqual(alignment_gate(20, "a$_$1", candidates, decisions), "BLOCKED")


if __name__ == "__main__":
    unittest.main()
