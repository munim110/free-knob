"""Cache the persistence forecast in the same layout as every other arm.

Persistence repeats the last observed frame of the input sequence for all twelve
lead times. It carries no parameters and needs no checkpoint, but it is written
to an HDF5 like the trained arms so that one pipeline scores all of them and no
arm gets a code path of its own.

Its role in the paper is as a control on the mechanism: it is a real radar field,
so max-pooling lifts it and the observation together and its pooled frequency
bias should sit at one whatever the threshold.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

import sys as _s2
from pathlib import Path as _P2
_s2.path.insert(0, str(_P2(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_pred_persistence.h5"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    P.require(args.h5, "SEVIR nowcast split")
    with h5py.File(args.h5, "r") as fg, h5py.File(args.out, "w") as fo:
        IN = fg["IN_vil"]
        n, h, w = IN.shape[0], IN.shape[1], IN.shape[2]
        t_out = fg["OUT_vil"].shape[-1]
        d = fo.create_dataset("pred_vil", shape=(n, h, w, t_out),
                              dtype="float32", chunks=(1, h, w, t_out),
                              compression="lzf")
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            last = IN[s:e, :, :, -1].astype(np.float32)
            d[s:e] = np.repeat(last[:, :, :, None], t_out, axis=3)
            print(f"  {e}/{n}", flush=True)
        fo.attrs["model"] = "persistence"
        fo.attrs["source"] = str(args.h5)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
