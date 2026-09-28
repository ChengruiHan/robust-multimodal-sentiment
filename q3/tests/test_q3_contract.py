import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.q3.core import (COALITIONS, contribution_summary, explain_modalities,
                         explain_windows, local_windows, masks_for_coalitions,
                         position_importance, select_evidence, shapley)
from src.q3.run import fidelity, main_row, read_attachment, scores
from src.q3.trace import locate, words_with_spans


class FakePredictor:
    def predict_masks(self, sample, masks, batch_size=64):
        # Each observed modality has a distinct effect; no neural dependency.
        raw = sample["raw_available"][0]
        keep = np.array([(raw & ~m).sum(axis=1) for m in masks], dtype=float)
        logits = np.column_stack((np.zeros(len(masks)),
                                  np.ones(len(masks)) * -.5,
                                  .2 * keep[:, 0] + .3 * keep[:, 1] + .1 * keep[:, 2]))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True), .1 * keep[:, 0] - .2 * keep[:, 1]


def sample():
    bert = np.zeros((1, 3, 50), dtype=np.int64)
    bert[0, 0, :6] = [101, 1000, 1001, 1002, 1003, 102]
    bert[0, 1, :6] = 1
    content = np.zeros((1, 50), dtype=bool)
    content[0, 1:5] = True
    raw = np.repeat(content[:, None], 3, axis=1)
    return dict(text_bert=bert, audio=np.ones((1, 50, 74), dtype=np.float32),
                vision=np.ones((1, 50, 35), dtype=np.float32),
                content=content, raw_available=raw)


class Q3Contract(unittest.TestCase):
    def test_shapley_additive_game(self):
        coefficients = np.array([2., -1., 3.])
        values = np.array([sum(coefficients[list(s)]) if s else 0 for s in COALITIONS])
        np.testing.assert_allclose(shapley(values), coefficients)

    def test_coalition_and_window_masks(self):
        s = sample()
        masks = masks_for_coalitions(s)
        self.assertEqual(masks.shape, (8, 3, 50))
        self.assertTrue(masks[0, :, 1:5].all())
        self.assertFalse(masks[-1].any())
        local, windows = local_windows(s, 3)
        self.assertEqual(len(windows), 6)
        self.assertTrue(all(1 <= a < b <= 5 for _, a, b in windows))
        self.assertFalse(local[:, :, 0].any())
        self.assertFalse(local[:, :, 5:].any())

    def test_explanation_and_export_contract(self):
        s = sample()
        predictor = FakePredictor()
        modal = explain_modalities(predictor, s)
        self.assertEqual(modal["target"], 2)
        self.assertAlmostEqual(sum(modal["phi_cls"]), modal["values_cls"][-1] - modal["values_cls"][0])
        windows = explain_windows(predictor, s, modal)
        self.assertEqual(position_importance(windows).shape, (3, 50))
        evidence = select_evidence(windows)
        self.assertGreater(len(evidence), 0)
        faith = fidelity(predictor, s, modal, evidence, repeats=5)
        self.assertGreater(faith["steps"], 0)
        self.assertTrue(np.isfinite(faith["aopc"]))
        row = main_row("01", "fake", modal, "config", "unresolved")
        self.assertAlmostEqual(sum(row[k] for k in ("share_text", "share_audio", "share_vision")), 1)
        self.assertEqual(row["pred_label"], "Positive")

    def test_zero_attribution(self):
        self.assertEqual(contribution_summary(np.zeros(3))["primary"], "undetermined")

    def test_metrics(self):
        result = scores(np.array([0, 1, 2]), np.array([0, 2, 2]),
                        np.array([-1., 0., 1.]), np.array([-1., .3, .8]))
        self.assertEqual(result["confusion"], [[1, 0, 0], [0, 0, 1], [0, 0, 1]])

    def test_unresolved_location_never_invents_time(self):
        trace = dict(raw_text="happy", tokens=[dict(position=1, word_id=0, char_start=0, char_end=5)],
                     words=words_with_spans("happy"), starts=np.array([np.nan]),
                     ends=np.array([np.nan]), statuses=["alignment_not_run"],
                     pts=np.empty(0), duration=3.)
        result = locate(trace, dict(modality="audio", start_idx=1, end_idx=2))
        self.assertEqual(result["text_span"], "happy")
        self.assertIsNone(result["start_sec"])
        self.assertEqual(result["mapping_quality"], "unresolved")

    def test_attachment4_real_schema(self):
        path = ROOT.parent.parent / "E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本/01.pkl"
        if path.exists():
            item, text = read_attachment(path)
            self.assertEqual(item["id"][0], "01")
            self.assertIn("Replacing", text)


if __name__ == "__main__":
    unittest.main()
