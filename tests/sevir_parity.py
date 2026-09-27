"""Real-data parity check against the paper's shipped CasCast control.

This is intentionally separate from the fast unit suite. It needs the official
SEVIR nowcast HDF5 and one cached prediction HDF5, reads them in bounded batches,
and compares the public FreeKnob primitives with all 30 raw/global-qmap cells in
the corresponding shipped JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

from freeknob import contingency_score, fit_quantile_map, spatial_pool


THRESHOLDS = [16, 74, 133, 160, 181, 219]
POOLINGS = {
    "none": ("none", 1),
    "avg4": ("mean", 4),
    "max4": ("max", 4),
    "avg16": ("mean", 16),
    "max16": ("max", 16),
}
FIT_BUDGET = 60_000_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-7)
    return parser


def _add(target: list[int], score) -> None:
    target[0] += score.hits
    target[1] += score.false_alarms
    target[2] += score.misses


def _metrics(counts: list[int]) -> tuple[float, float]:
    hits, false_alarms, misses = counts
    return (
        hits / max(hits + false_alarms + misses, 1),
        (hits + false_alarms) / max(hits + misses, 1),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    started = time.time()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    gold_rows = {
        (row["pooling"], row["threshold"]): row for row in gold["table"]
    }

    with h5py.File(args.pred, "r") as pred_file, h5py.File(args.gt, "r") as gt_file:
        predictions = pred_file["pred_vil"]
        observations = gt_file["OUT_vil"]
        if predictions.shape != observations.shape:
            raise ValueError(
                f"HDF5 shapes differ: {predictions.shape} != {observations.shape}"
            )
        n_events = predictions.shape[0]
        calibration = np.arange(0, n_events, 2)
        evaluation = np.arange(1, n_events, 2)
        if len(calibration) != gold["n_calib"] or len(evaluation) != gold["n_eval"]:
            raise ValueError("split sizes do not match the golden artefact")

        values_per_event = int(np.prod(predictions.shape[1:]))
        fit_events = max(1, FIT_BUDGET // values_per_event)
        stride = max(1, len(calibration) // fit_events)
        fit_indices = calibration[::stride][:fit_events]
        print(
            f"fit: {len(fit_indices)} events, "
            f"{len(fit_indices) * values_per_event:,} values per marginal"
        )
        fit_predictions = predictions[fit_indices]
        fit_observations = observations[fit_indices]
        calibrator = fit_quantile_map(
            fit_predictions,
            fit_observations,
            n_quantiles=1001,
            max_samples=None,
        )
        del fit_predictions, fit_observations

        counts = {
            pooling: {
                mode: {threshold: [0, 0, 0] for threshold in THRESHOLDS}
                for mode in ("raw", "calibrated")
            }
            for pooling in POOLINGS
        }
        for start in range(0, len(evaluation), args.batch_size):
            indices = evaluation[start:start + args.batch_size]
            pred = np.moveaxis(predictions[indices].astype(np.float32), -1, 1)
            obs = np.moveaxis(observations[indices].astype(np.float32), -1, 1)
            calibrated = calibrator(pred)
            for label, (pooling, block_size) in POOLINGS.items():
                pooled_obs = spatial_pool(obs, pooling, block_size)
                pooled_raw = spatial_pool(pred, pooling, block_size)
                pooled_calibrated = spatial_pool(
                    calibrated, pooling, block_size
                )
                for threshold in THRESHOLDS:
                    _add(
                        counts[label]["raw"][threshold],
                        contingency_score(pooled_raw, pooled_obs, threshold),
                    )
                    _add(
                        counts[label]["calibrated"][threshold],
                        contingency_score(
                            pooled_calibrated, pooled_obs, threshold
                        ),
                    )
            completed = min(start + args.batch_size, len(evaluation))
            if completed % 100 == 0 or completed == len(evaluation):
                print(f"score: {completed}/{len(evaluation)} evaluation events")

    rows = []
    failures = []
    for pooling in POOLINGS:
        for threshold in THRESHOLDS:
            raw_csi, raw_bias = _metrics(counts[pooling]["raw"][threshold])
            cal_csi, cal_bias = _metrics(
                counts[pooling]["calibrated"][threshold]
            )
            expected = gold_rows[(pooling, threshold)]
            differences = {
                "csi_uncal": raw_csi - expected["csi_uncal"],
                "bias_uncal": raw_bias - expected["bias_uncal"],
                "csi_qmap_global": cal_csi - expected["csi_qmap_global"],
                "bias_qmap_global": cal_bias - expected["bias_qmap_global"],
            }
            failed = {
                key: value for key, value in differences.items()
                if abs(value) > args.atol
            }
            if failed:
                failures.append({
                    "pooling": pooling,
                    "threshold": threshold,
                    "differences": failed,
                })
            rows.append({
                "pooling": pooling,
                "threshold": threshold,
                "raw_counts": counts[pooling]["raw"][threshold],
                "calibrated_counts": counts[pooling]["calibrated"][threshold],
                "csi_uncal": raw_csi,
                "bias_uncal": raw_bias,
                "csi_qmap_global": cal_csi,
                "bias_qmap_global": cal_bias,
                "differences": differences,
            })

    result = {
        "protocol": "FreeKnob Audit real-data parity",
        "prediction": args.pred.name,
        "ground_truth": args.gt.name,
        "golden": args.gold.name,
        "n_calibration": len(calibration),
        "n_evaluation": len(evaluation),
        "n_fit_events": len(fit_indices),
        "fit_values_per_marginal": len(fit_indices) * values_per_event,
        "absolute_tolerance": args.atol,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if result['passed'] else 'FAIL'}: "
        f"{len(rows) - len(failures)}/{len(rows)} cells within {args.atol:g}; "
        f"wrote {args.out}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

