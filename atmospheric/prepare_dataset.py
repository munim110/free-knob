"""Build a band's chronological train/val/test split and its normalisation stats.

Reads the per-timestamp `<stamp>_predictor.npy` and `<stamp>_target.npy` pairs
produced from the AR event list, splits them 80/10/10 in time, fits the
normalisation statistics on the training portion alone, and writes the layout
`common/data/dataset.py` expects:

    <out-dir>/train/       <stamp>_predictor.npy, <stamp>_target.npy
    <out-dir>/val/
    <out-dir>/test/
    <out-dir>/normalization_stats.joblib

with `<out-dir>` defaulting to `$ATMOS_DATA/B<band>`, the path the training
scripts and REPRODUCE.md name.

The split is strictly chronological: the earliest 80% train, the next 10%
validate, the latest 10% test. Satellite scenes six hours apart are strongly
correlated, so a random split would leak across the boundary and inflate every
score. Because the split follows the calendar, the three portions also see
different weather: the validation portion of the released record is the
quietest of the three, and the knob each arm receives is selected on it.

Predictor statistics are accumulated independently per channel over training
pixels. This is the same estimator used to create the archived statistics for
the scored checkpoints: population mean and standard deviation, fitted on the
training portion only. Target statistics use the same population convention.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402

VAL_SIZE, TEST_SIZE = 0.10, 0.10


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="directory of <stamp>_predictor.npy / _target.npy pairs")
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: $ATMOS_DATA/B<band>")
    ap.add_argument("--move", action="store_true",
                    help="move rather than copy, for large records")
    return ap.parse_args()


def chronological_split(stamps):
    """Split sorted timestamps 80/10/10 by position, latest last."""
    n = len(stamps)
    n_test = int(n * TEST_SIZE)
    n_val = int(n * VAL_SIZE)
    return {
        "train": stamps[:n - n_test - n_val],
        "val": stamps[n - n_test - n_val:n - n_test],
        "test": stamps[n - n_test:],
    }


def stats_per_channel(paths, n_channels):
    """Population mean/std per channel, matching the archived estimator."""
    count = 0
    total = np.zeros(n_channels, dtype=np.float64)
    total_sq = np.zeros(n_channels, dtype=np.float64)
    for p in paths:
        x = np.load(p).reshape(n_channels, -1).astype(np.float64)
        total += x.sum(axis=1)
        total_sq += (x * x).sum(axis=1)
        count += x.shape[1]
    mean = total / count
    variance = np.maximum(total_sq / count - mean ** 2, 0.0)
    return mean, np.sqrt(variance)


def main():
    args = parse_args()

    # Imported after argument parsing so that --help works in a numpy-only
    # environment. joblib writes the stats file, and the dataset module pulls
    # in torch, which this script itself does not need.
    import joblib
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common" / "data"))
    from dataset import ALL_VARIABLES

    P.require(args.processed_dir, "directory of processed npy pairs")
    out_dir = args.out_dir or Path(str(P.ATMOS_DATA) + f"/B{args.band:02d}")

    stamps = sorted(p.name[:-len("_predictor.npy")]
                    for p in args.processed_dir.glob("*_predictor.npy"))
    if not stamps:
        raise SystemExit(f"no *_predictor.npy pairs under {args.processed_dir}")
    splits = chronological_split(stamps)

    print(f"band B{args.band:02d}, {len(stamps)} paired scenes")
    for name, part in splits.items():
        span = f"{part[0]} to {part[-1]}" if part else "empty"
        print(f"  {name:5s} {len(part):5d}   {span}")

    train_pred = [args.processed_dir / f"{s}_predictor.npy"
                  for s in splits["train"]]
    n_channels = np.load(train_pred[0]).shape[0]

    print(f"\nfitting predictor statistics on the training portion "
          f"(stats_per_channel, {n_channels} channels)")
    pred_mean, pred_std = stats_per_channel(train_pred, n_channels)

    targets = np.concatenate([
        np.load(args.processed_dir / f"{s}_target.npy").ravel()
        for s in splits["train"]])
    stats = {
        "predictor_mean": pred_mean,
        "predictor_std": pred_std,
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "variables": list(ALL_VARIABLES[:n_channels]),
        "stats_estimator": "per_channel_population",
    }
    for i, v in enumerate(stats["variables"]):
        print(f"  {v:6s} mean {pred_mean[i]:12.4f}   std {pred_std[i]:12.4f}")
    print(f"  target mean {stats['target_mean']:.4f}   "
          f"std {stats['target_std']:.4f}")

    place = shutil.move if args.move else shutil.copy
    for name, part in splits.items():
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        for s in part:
            for kind in ("predictor", "target"):
                src = args.processed_dir / f"{s}_{kind}.npy"
                if src.exists():
                    place(str(src), str(dest / f"{s}_{kind}.npy"))
        print(f"placed {len(part)} scenes in {dest}")

    joblib.dump(stats, out_dir / "normalization_stats.joblib")
    print(f"\nwrote {out_dir / 'normalization_stats.joblib'}")
    print(f"Train with --data-dir {out_dir}")


if __name__ == "__main__":
    main()
