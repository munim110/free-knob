import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from freeknob import (
    contingency_score,
    fit_quantile_map,
    free_knob_audit,
    spatial_pool,
    threshold_pool_score,
)


class QuantileMapTests(unittest.TestCase):
    def test_linear_marginals_map_linearly(self):
        calibrator = fit_quantile_map(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 10.0, 20.0, 30.0]),
            n_quantiles=101,
        )
        np.testing.assert_allclose(calibrator([0.5, 1.5, 2.5]), [5, 15, 25])

    def test_map_is_monotone(self):
        calibrator = fit_quantile_map(
            np.array([0, 0, 0, 1, 2, 2]),
            np.array([0, 1, 1, 3, 4, 5]),
            n_quantiles=51,
        )
        mapped = calibrator(np.linspace(-1, 3, 100))
        self.assertTrue(np.all(np.diff(mapped) >= 0))

    def test_fit_ignores_nonfinite_calibration_values(self):
        calibrator = fit_quantile_map(
            np.array([0.0, 1.0, np.nan, np.inf]),
            np.array([0.0, 2.0, np.nan, -np.inf]),
            n_quantiles=11,
        )
        np.testing.assert_allclose(calibrator([0.25, 0.75]), [0.5, 1.5])

    def test_fit_rejects_empty_or_all_nonfinite_input(self):
        for predictions in (np.array([]), np.array([np.nan, np.inf])):
            with self.subTest(predictions=predictions):
                with self.assertRaisesRegex(ValueError, "finite value"):
                    fit_quantile_map(predictions, np.array([0.0, 1.0]))

    def test_extrapolation_uses_end_segments(self):
        calibrator = fit_quantile_map(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 4.0]),
            n_quantiles=21,
        )
        np.testing.assert_allclose(calibrator([-1.0, 3.0]), [-2.0, 6.0])

    def test_random_maps_preserve_order(self):
        rng = np.random.default_rng(7)
        for trial in range(25):
            with self.subTest(trial=trial):
                pred = rng.normal(size=200)
                obs = rng.gamma(shape=2.0, size=200)
                calibrator = fit_quantile_map(pred, obs, n_quantiles=101)
                ordered = np.sort(rng.normal(size=300))
                self.assertTrue(np.all(np.diff(calibrator(ordered)) >= 0))

    def test_sampling_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(3)
        pred, obs = rng.normal(size=1000), rng.normal(size=1000)
        first = fit_quantile_map(
            pred, obs, max_samples=100, n_quantiles=51, seed=19
        )
        second = fit_quantile_map(
            pred, obs, max_samples=100, n_quantiles=51, seed=19
        )
        np.testing.assert_array_equal(
            first.source_quantiles, second.source_quantiles
        )
        np.testing.assert_array_equal(
            first.target_quantiles, second.target_quantiles
        )

    def test_reference_map_uses_float32_precision(self):
        calibrator = fit_quantile_map(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 4.0]),
            n_quantiles=11,
        )
        self.assertEqual(calibrator.source_quantiles.dtype, np.float32)
        self.assertEqual(calibrator.target_quantiles.dtype, np.float32)
        self.assertEqual(calibrator(np.array([0.5], dtype=np.float32)).dtype,
                         np.float32)


class ScoringTests(unittest.TestCase):
    def test_mean_and_max_pool_continuous_values_before_threshold(self):
        field = np.array([[[0.0, 0.0], [0.0, 4.0]]])
        mean = threshold_pool_score(
            field, field, 2.0, pooling="mean", block_size=2
        )
        maximum = threshold_pool_score(
            field, field, 2.0, pooling="max", block_size=2
        )
        self.assertEqual(mean.hits, 0)
        self.assertEqual(maximum.hits, 1)

    def test_nondivisible_edges_are_dropped(self):
        field = np.arange(25).reshape(5, 5)
        pooled = spatial_pool(field, "max", 2)
        np.testing.assert_array_equal(pooled, [[6, 8], [16, 18]])

    def test_pooling_aliases_match(self):
        field = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(
            spatial_pool(field, "avg", 2), spatial_pool(field, "mean", 2)
        )
        np.testing.assert_array_equal(
            spatial_pool(field, "identity", 1), field
        )

    def test_pooling_preserves_float32_precision(self):
        field = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.assertEqual(spatial_pool(field, "none", 1).dtype, np.float32)
        self.assertEqual(spatial_pool(field, "mean", 2).dtype, np.float32)
        self.assertEqual(spatial_pool(field, "max", 2).dtype, np.float32)

    def test_invalid_pooling_and_block_size_are_rejected(self):
        field = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "pooling"):
            spatial_pool(field, "median", 2)
        for block_size in (0, -1, 3):
            with self.subTest(block_size=block_size):
                with self.assertRaises(ValueError):
                    spatial_pool(field, "max", block_size)

    def test_score_counts_and_derived_metrics(self):
        score = contingency_score(
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 1.0, 0.0]),
            threshold=0.5,
        )
        self.assertEqual((score.hits, score.false_alarms, score.misses), (1, 1, 1))
        self.assertAlmostEqual(score.csi, 1 / 3)
        self.assertEqual(score.frequency_bias, 1.0)
        self.assertEqual(score.deviation, 0.0)

    def test_zero_forecasts_have_undefined_log_deviation(self):
        score = contingency_score([0.0, 0.0], [1.0, 0.0], threshold=0.5)
        self.assertEqual(score.frequency_bias, 0.0)
        self.assertIsNone(score.deviation)

    def test_evaluation_rejects_empty_nonfinite_and_mismatched_arrays(self):
        cases = [
            (np.array([]), np.array([])),
            (np.array([np.nan]), np.array([0.0])),
            (np.array([0.0]), np.array([0.0, 1.0])),
        ]
        for pred, obs in cases:
            with self.subTest(pred=pred, obs=obs):
                with self.assertRaises(ValueError):
                    contingency_score(pred, obs, threshold=0.5)

    def test_random_pooling_matches_a_loop_reference(self):
        rng = np.random.default_rng(11)
        for pooling in ("mean", "max"):
            for block_size in (2, 3, 4):
                field = rng.normal(size=(2, 11, 13))
                height = field.shape[-2] // block_size
                width = field.shape[-1] // block_size
                reference = np.empty((2, height, width))
                for event in range(2):
                    for row in range(height):
                        for column in range(width):
                            block = field[
                                event,
                                row * block_size:(row + 1) * block_size,
                                column * block_size:(column + 1) * block_size,
                            ]
                            reference[event, row, column] = (
                                block.mean() if pooling == "mean" else block.max()
                            )
                np.testing.assert_allclose(
                    spatial_pool(field, pooling, block_size), reference
                )

    def test_random_contingency_counts_match_boolean_reference(self):
        rng = np.random.default_rng(17)
        for trial in range(25):
            with self.subTest(trial=trial):
                pred = rng.normal(size=(7, 9))
                obs = rng.normal(size=(7, 9))
                threshold = float(rng.normal())
                pred_positive, obs_positive = pred >= threshold, obs >= threshold
                score = contingency_score(pred, obs, threshold)
                self.assertEqual(
                    (score.hits, score.false_alarms, score.misses),
                    (
                        int(np.count_nonzero(pred_positive & obs_positive)),
                        int(np.count_nonzero(pred_positive & ~obs_positive)),
                        int(np.count_nonzero(~pred_positive & obs_positive)),
                    ),
                )


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.obs_cal = np.array([0.0, 1.0, 2.0, 3.0])
        self.obs_test = np.array([0.0, 0.0, 3.0, 3.0, 3.0, 3.0])
        self.pred_cal = {
            "attenuated": self.obs_cal * 0.1,
            "sharp": self.obs_cal.copy(),
        }
        self.pred_test = {
            "attenuated": np.array([0.0, 0.0, 0.3, 0.3, 3.0, 0.3]),
            "sharp": np.array([0.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
        }

    def test_symmetric_control_can_reverse_a_ranking(self):
        report = free_knob_audit(
            self.pred_cal,
            self.obs_cal,
            self.pred_test,
            self.obs_test,
            threshold=2.0,
        )
        pair = report.pairs[0]
        self.assertGreater(pair.raw_contrast, 0)
        self.assertLess(pair.calibrated_contrast, 0)
        self.assertTrue(pair.ranking_reversal)
        self.assertGreater(pair.calibration_gap, 0)
        self.assertAlmostEqual(
            pair.distortion, pair.raw_contrast - pair.calibrated_contrast
        )

    def test_max_pooling_precedes_thresholding(self):
        obs_cal = np.array([[[0.0, 0.0], [0.0, 3.0]]])
        report = free_knob_audit(
            {"arm": obs_cal},
            obs_cal,
            {"arm": obs_cal},
            obs_cal,
            threshold=2.0,
            pooling="max",
            block_size=2,
        )
        self.assertEqual(report.arms["arm"].raw.hits, 1)
        self.assertEqual(report.arms["arm"].raw.csi, 1.0)

    def test_calibration_and_test_arms_must_match(self):
        with self.assertRaisesRegex(ValueError, "same arms"):
            free_knob_audit(
                {"a": self.obs_cal}, self.obs_cal,
                {"b": self.obs_test}, self.obs_test,
                threshold=2.0,
            )

    def test_three_arms_produce_all_three_pairs_in_input_order(self):
        predictions_calibration = {
            "first": self.obs_cal,
            "second": self.obs_cal * 0.5,
            "third": self.obs_cal * 2.0,
        }
        predictions_test = {
            "first": self.obs_test,
            "second": self.obs_test * 0.5,
            "third": self.obs_test * 2.0,
        }
        report = free_knob_audit(
            predictions_calibration,
            self.obs_cal,
            predictions_test,
            self.obs_test,
            threshold=2.0,
        )
        self.assertEqual(
            [(pair.arm_a, pair.arm_b) for pair in report.pairs],
            [("first", "second"), ("first", "third"), ("second", "third")],
        )

    def test_serialized_calibrators_are_optional(self):
        report = free_knob_audit(
            {"arm": self.obs_cal}, self.obs_cal,
            {"arm": self.obs_test}, self.obs_test,
            threshold=2.0,
        )
        self.assertNotIn("calibrators", report.to_dict())
        self.assertIn("calibrators", report.to_dict(include_calibrators=True))

    def test_each_arm_must_match_its_split_shape(self):
        with self.assertRaisesRegex(ValueError, "calibration shape"):
            free_knob_audit(
                {"arm": self.obs_cal[:-1]}, self.obs_cal,
                {"arm": self.obs_test}, self.obs_test,
                threshold=2.0,
            )
        with self.assertRaisesRegex(ValueError, "test shape"):
            free_knob_audit(
                {"arm": self.obs_cal}, self.obs_cal,
                {"arm": self.obs_test[:-1]}, self.obs_test,
                threshold=2.0,
            )

    def test_cli_reads_npz_and_emits_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.npz"
            np.savez(
                path,
                obs_calibration=self.obs_cal,
                obs_test=self.obs_test,
                attenuated__calibration=self.pred_cal["attenuated"],
                attenuated__test=self.pred_test["attenuated"],
                sharp__calibration=self.pred_cal["sharp"],
                sharp__test=self.pred_test["sharp"],
            )
            run = subprocess.run(
                [sys.executable, "-m", "freeknob", str(path),
                 "--threshold", "2"],
                check=True,
                capture_output=True,
                text=True,
            )
            output = json.loads(run.stdout)
            self.assertEqual(output["protocol"], "FreeKnob Audit")
            self.assertTrue(output["pairs"][0]["ranking_reversal"])

    def test_cli_rejects_an_npz_without_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(path, arm__calibration=np.array([0.0]))
            run = subprocess.run(
                [sys.executable, "-m", "freeknob", str(path),
                 "--threshold", "2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 2)
            self.assertIn("missing NPZ arrays", run.stderr)


if __name__ == "__main__":
    unittest.main()
