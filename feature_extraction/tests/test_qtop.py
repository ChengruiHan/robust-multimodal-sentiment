import unittest

import numpy as np

from scripts.qtop import FrameSeries, aggregate, support_cells
from scripts.q1_sources import official_words, safe_id


class QtopTest(unittest.TestCase):
    def test_support_and_weighted_statistics(self):
        series = FrameSeries(np.array([[1.0], [3.0], [99.0]]),
                             np.array([0.25, 0.75, 1.25]),
                             np.array([1.0, 0.5, 0.0]),
                             np.array([10, 12, 13]), 0.0, 1.5)
        np.testing.assert_allclose(support_cells(series.centers, 0, 1.5),
                                   [[0, 0.5], [0.5, 1.0], [1.0, 1.5]])
        result = aggregate(np.array([0.25, 1.0, np.nan]),
                           np.array([0.75, 1.5, np.nan]),
                           np.array([True, True, False]), series, 1)
        # First word has 0.25 s of each of two cells, weighted 1 : 0.5.
        np.testing.assert_allclose(result["features"][0], [5 / 3, np.sqrt(8 / 9)], rtol=1e-6)
        self.assertEqual(result["coverage"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result["available"].tolist(), [True, False, False])
        self.assertEqual(result["dispersion_valid"].tolist(), [True, False, False])
        self.assertEqual(result["frame_offsets"].tolist(), [0, 2, 2, 2])
        self.assertEqual(result["frame_ids"].tolist(), [10, 12])
        self.assertAlmostEqual(result["quality"][0], 0.75)

    def test_single_cell_and_no_overlap(self):
        series = FrameSeries(np.array([[2.0]]), np.array([0.5]), np.array([1.0]),
                             np.array([4]), 0.0, 1.0)
        result = aggregate(np.array([0.2, 1.1]), np.array([0.4, 1.3]),
                           np.array([True, True]), series, 1)
        np.testing.assert_array_equal(result["features"], [[2, 0], [0, 0]])
        self.assertFalse(result["dispersion_valid"][0])
        self.assertFalse(result["available"][1])

    def test_text_spans_and_filename(self):
        raw = "They've been, fine!"
        words = official_words(raw)
        self.assertEqual([raw[w["char_start"]:w["char_end"]] for w in words],
                         ["They've", "been", "fine"])
        self.assertNotEqual(safe_id("a", "bc"), safe_id("ab", "c"))


if __name__ == "__main__":
    unittest.main()
