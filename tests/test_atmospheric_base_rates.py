import tempfile
import unittest
from pathlib import Path

import numpy as np

from atmospheric.base_rates import audit_split, crop_valid


class AtmosphericBaseRateTests(unittest.TestCase):
    def test_crop_valid_matches_centered_evaluation_support(self):
        array = np.arange(2 * 7 * 5).reshape(2, 7, 5)
        cropped = crop_valid(array, (4, 6))
        np.testing.assert_array_equal(cropped, array[:, 1:5, :])

    def test_audit_names_unconditional_and_conditioned_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            first = np.zeros((4, 3), dtype=np.float32)
            second = np.full((4, 3), 300.0, dtype=np.float32)
            np.save(split / "20200101_0000_target.npy", first)
            np.save(split / "20200101_0600_target.npy", second)

            result = audit_split(split, (220.0,), (2, 4), 0.1)
            row = result["thresholds"]["220"]
            self.assertEqual(result["full_grid_shape"], [4, 3])
            self.assertEqual(result["evaluation_valid_shape"], [2, 3])
            self.assertEqual(row["n_frames"], 2)
            self.assertEqual(row["n_eventful_frames"], 1)
            self.assertEqual(row["full_grid_all_frames"], 0.5)
            self.assertEqual(row["evaluation_footprint_all_frames"], 0.5)
            self.assertEqual(row["evaluation_footprint_eventful_frames"], 1.0)


if __name__ == "__main__":
    unittest.main()
