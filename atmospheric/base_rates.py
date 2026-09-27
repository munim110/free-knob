"""Audit atmospheric positive fractions under every protocol definition.

The atmospheric record is stored on a 351 x 251 geographic grid, while model
evaluation center-crops the latitude dimension to 256 and pads the five missing
columns of a nominal 256 x 256 tensor behind a validity mask.  In addition, the
historical score excludes frames whose observed critical coverage is at most a
configured percentage.  Those choices create several valid but distinct base
rates.  This script names all of them and writes the two compact compatibility
files consumed by the existing paper generators.

No predictions or checkpoints are needed.  Only ``*_target.npy`` files under
``<data-dir>/{train,val,test}`` are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_THRESHOLDS = (220.0, 215.0, 210.0, 205.0, 200.0, 197.0)


def parse_shape(value: str) -> tuple[int, int]:
    parts = tuple(int(item) for item in value.lower().split("x"))
    if len(parts) != 2 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("shape must be HxW with positive integers")
    return parts


def parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS,
        help="comma-separated Kelvin thresholds (default: 220,215,210,205,200,197)",
    )
    parser.add_argument("--eval-shape", type=parse_shape, default=(256, 256))
    parser.add_argument(
        "--eventful-min-percent", type=float, default=0.1,
        help="historical strict frame filter, observed coverage > this percentage",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--r-test-out", type=Path, default=None,
        help="optional compatibility JSON: test/evaluation-footprint/all-frame rates",
    )
    parser.add_argument(
        "--r-clim-out", type=Path, default=None,
        help="optional compatibility JSON: train/evaluation-footprint/eventful rates",
    )
    return parser.parse_args()


def crop_valid(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Return only valid pixels from the evaluator's centered crop/pad geometry."""
    height, width = array.shape[-2:]
    target_h, target_w = target_shape
    start_h = max(0, (height - target_h) // 2)
    start_w = max(0, (width - target_w) // 2)
    end_h = min(height, start_h + target_h)
    end_w = min(width, start_w + target_w)
    return array[..., start_h:end_h, start_w:end_w]


def threshold_rates(
    full: np.ndarray,
    evaluated: np.ndarray,
    threshold: float,
    eventful_min_percent: float,
) -> dict[str, float | int]:
    full_per_frame = np.mean(full <= threshold, axis=(1, 2))
    eval_per_frame = np.mean(evaluated <= threshold, axis=(1, 2))
    eventful = eval_per_frame > eventful_min_percent / 100.0
    if not np.any(eventful):
        conditioned = None
    else:
        # Every frame has the same valid support, so the mean of per-frame
        # fractions is exactly the pooled fraction over the selected cohort.
        conditioned = float(np.mean(eval_per_frame[eventful]))
    return {
        "full_grid_all_frames": float(np.mean(full_per_frame)),
        "evaluation_footprint_all_frames": float(np.mean(eval_per_frame)),
        "evaluation_footprint_eventful_frames": conditioned,
        "n_frames": int(len(eval_per_frame)),
        "n_eventful_frames": int(np.sum(eventful)),
    }


def audit_split(
    split_dir: Path,
    thresholds: tuple[float, ...],
    eval_shape: tuple[int, int],
    eventful_min_percent: float,
) -> dict[str, object]:
    files = sorted(split_dir.glob("*_target.npy"))
    if not files:
        raise FileNotFoundError(f"no *_target.npy files under {split_dir}")
    full = np.stack([np.load(path) for path in files])
    if full.ndim != 3:
        raise ValueError(f"targets must stack to (N,H,W), got {full.shape}")
    evaluated = crop_valid(full, eval_shape)
    return {
        "first_timestamp": files[0].name.removesuffix("_target.npy"),
        "last_timestamp": files[-1].name.removesuffix("_target.npy"),
        "full_grid_shape": list(full.shape[1:]),
        "evaluation_valid_shape": list(evaluated.shape[1:]),
        "thresholds": {
            f"{threshold:g}": threshold_rates(
                full, evaluated, threshold, eventful_min_percent
            )
            for threshold in thresholds
        },
    }


def compatibility_table(
    report: dict[str, object], split: str, field: str
) -> dict[str, float]:
    rows = report["splits"][split]["thresholds"]
    return {
        key: value[field]
        for key, value in rows.items()
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = {
        "schema_version": 1,
        "event_definition": "target_brightness_temperature_k <= threshold_k",
        "evaluation_shape": list(args.eval_shape),
        "eventful_filter": {
            "field": "evaluation_footprint_observed_coverage_percent",
            "operator": ">",
            "value": args.eventful_min_percent,
        },
        "definitions": {
            "paper_R": "test/evaluation_footprint/all_frames",
            "selection_rule_D_climatology": (
                "train/evaluation_footprint/eventful_frames"
            ),
        },
        "splits": {
            split: audit_split(
                args.data_dir / split,
                args.thresholds,
                args.eval_shape,
                args.eventful_min_percent,
            )
            for split in ("train", "val", "test")
        },
    }
    write_json(args.out, report)

    if args.r_test_out is not None:
        write_json(
            args.r_test_out,
            compatibility_table(
                report, "test", "evaluation_footprint_all_frames"
            ),
        )
    if args.r_clim_out is not None:
        write_json(
            args.r_clim_out,
            compatibility_table(
                report, "train", "evaluation_footprint_eventful_frames"
            ),
        )

    row = report["splits"]["test"]["thresholds"]["220"]
    print(
        "wrote", args.out,
        f"(tau=220: full={100 * row['full_grid_all_frames']:.4f}%, "
        f"evaluation={100 * row['evaluation_footprint_all_frames']:.4f}%)",
    )


if __name__ == "__main__":
    main()
