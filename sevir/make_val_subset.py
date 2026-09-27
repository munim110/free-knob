"""Carve a validation split out of the SEVIR nowcast training file, by date.

Everywhere else in this paper the calibration knob is fitted on one index-parity
half of the test split and evaluated on the other. That is clean and symmetric
for the confound claim, since both halves come from one distribution and the
comparison between arms is what is at stake. It is not the setting a deployed
system faces, where the knob would be fitted on data from before the evaluation
period and applied across whatever drift lies between.

SEVIR's preprocessed nowcast release ships a training file and a testing file and
no separate validation file, so the split both models we decompose actually
trained against has to be reconstructed from the metadata. EarthFormer's loader
takes `train_val_split_date = 2019-01-01`; CasCast's file lists name their
validation events `SEVIR_VIL_RANDOMEVENTS_2019_0101_0430`. Both therefore hold
out the same window, 2019-01-01 to 2019-04-30, from training. The test file
begins on 2019-06-11, so that window is held out from both models AND lies
entirely before the evaluation period.

This script selects an evenly spaced subsample of that window and writes it in
the same layout as the test file, so the inference scripts run against it
unchanged.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import sys as _s2
from pathlib import Path as _P2
_s2.path.insert(0, str(_P2(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path,
                    default=P.SEVIR_DATA / "data/processed/nowcast_training_000.h5")
    ap.add_argument("--meta", type=Path,
                    default=P.SEVIR_DATA / "nowcast_training_000_META.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2019-05-01",
                    help="exclusive upper bound on event time")
    ap.add_argument("--n-events", type=int, default=128)
    ap.add_argument("--out", type=Path,
                    default=P.SEVIR_DATA / "nowcast_val_subset.h5")
    args = ap.parse_args()

    P.require(args.h5, "SEVIR nowcast training split")
    P.require(args.meta, "SEVIR nowcast training metadata")

    meta = pd.read_csv(args.meta)
    t = pd.to_datetime(meta["time_utc"])
    win = np.flatnonzero((t >= pd.Timestamp(args.start))
                         & (t < pd.Timestamp(args.end)))
    if len(win) == 0:
        raise SystemExit(f"no events in [{args.start}, {args.end})")
    step = max(1, len(win) // args.n_events)
    sel = np.sort(win[::step][:args.n_events])
    print(f"{len(win)} events in [{args.start}, {args.end}); "
          f"taking {len(sel)} evenly spaced")

    with h5py.File(args.h5, "r") as fi, h5py.File(args.out, "w") as fo:
        for key in ("IN_vil", "OUT_vil"):
            src = fi[key]
            dst = fo.create_dataset(key, shape=(len(sel),) + src.shape[1:],
                                    dtype=src.dtype,
                                    chunks=(1,) + src.shape[1:],
                                    compression="lzf")
            for j, i in enumerate(sel):
                dst[j] = src[i]
                if (j + 1) % 32 == 0:
                    print(f"  {key} {j + 1}/{len(sel)}", flush=True)
        fo.attrs["source"] = str(args.h5)
        fo.attrs["window"] = f"[{args.start}, {args.end})"
        fo.attrs["source_index"] = sel

    side = args.out.with_suffix(".json")
    side.write_text(json.dumps(
        {"source": str(args.h5), "window": [args.start, args.end],
         "n_in_window": int(len(win)), "n_taken": int(len(sel)),
         "source_index": [int(i) for i in sel],
         "time_utc": [str(x) for x in t.iloc[sel]]}, indent=2))
    print(f"wrote {args.out} and {side}")


if __name__ == "__main__":
    main()
