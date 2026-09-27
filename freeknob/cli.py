"""Command-line entry point for the FreeKnob Audit."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np

from .core import free_knob_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freeknob-audit",
        description=(
            "Fit one quantile map per arm on a calibration split and compare "
            "raw with calibrated pool-and-threshold CSI on a disjoint test split."
        ),
    )
    parser.add_argument(
        "data",
        type=Path,
        help=(
            "NPZ with obs_calibration, obs_test, and ARM__calibration / "
            "ARM__test arrays"
        ),
    )
    parser.add_argument("--arm", action="append", dest="arms",
                        help="arm to audit; repeat for several (default: infer all)")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--pooling", choices=("none", "mean", "max"),
                        default="none")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--n-quantiles", type=int, default=1001)
    parser.add_argument("--max-fit-samples", type=int, default=60_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-calibrators", action="store_true")
    parser.add_argument("--output", type=Path,
                        help="write JSON here instead of standard output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        context = np.load(args.data, allow_pickle=False)
        with context if hasattr(context, "__enter__") else nullcontext(context) as data:
            required = {"obs_calibration", "obs_test"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"missing NPZ arrays: {', '.join(sorted(missing))}")
            arms = args.arms or sorted(
                key[:-13] for key in data.files if key.endswith("__calibration")
            )
            if not arms:
                raise ValueError("no ARM__calibration arrays found")
            pred_cal = {arm: data[f"{arm}__calibration"] for arm in arms}
            pred_test = {arm: data[f"{arm}__test"] for arm in arms}
            report = free_knob_audit(
                pred_cal,
                data["obs_calibration"],
                pred_test,
                data["obs_test"],
                threshold=args.threshold,
                pooling=args.pooling,
                block_size=args.block_size,
                n_quantiles=args.n_quantiles,
                max_fit_samples=args.max_fit_samples,
                seed=args.seed,
            )
        text = report.to_json(
            args.output, include_calibrators=args.include_calibrators
        )
        if args.output is None:
            print(text)
    except (KeyError, OSError, ValueError) as error:
        print(f"freeknob-audit: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

