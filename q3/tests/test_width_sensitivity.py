"""Small arithmetic checks for width-comparison metrics."""
import unittest

import numpy as np

from src.q3.width_sensitivity import average_ranks, interval_stats, spearman


class WidthSensitivityMath(unittest.TestCase):
    def test_tied_ranks_and_opposite_order(self):
        np.testing.assert_allclose(average_ranks(np.array([1, 1, 3, 2])),
                                   [.5, .5, 3, 2])
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1], [1, 2]))

    def test_half_open_interval_overlap(self):
        result = interval_stats(dict(start_idx=1, end_idx=4),
                                dict(start_idx=3, end_idx=8))
        self.assertTrue(result["overlap"])
        self.assertAlmostEqual(result["iou"], 1 / 7)
        self.assertEqual(result["center_distance"], 3)
        touching = interval_stats(dict(start_idx=1, end_idx=4),
                                  dict(start_idx=4, end_idx=8))
        self.assertFalse(touching["overlap"])


if __name__ == "__main__":
    unittest.main()
