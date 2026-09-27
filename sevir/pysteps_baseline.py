"""pysteps optical-flow extrapolation on the SEVIR nowcast test split.

The classical baseline used in the DGMR and NowcastNet evaluations: estimate a
motion field from the input sequence with Lucas-Kanade, then advect the last
observed frame forward. Sharp by construction, so its frequency bias should sit
near 1 at every threshold. That is the contrast which isolates whether
recalibration gain tracks blur.

Runs on CPU across processes; the network models it is compared against ran on
GPU. No training privilege differs, and both are evaluated on the same events.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

H5 = Path(P.SEVIR_H5)
OUT_LEN = 12


def _forecast(args):
    idx, seq = args           # seq: (384,384,13) uint8
    from pysteps import motion, nowcasts
    x = seq.astype(np.float32).transpose(2, 0, 1)   # (13,H,W)
    # pysteps expects a rainfall-like field; VIL is monotone in intensity, and
    # every knob downstream is monotone, so working in native units is fine.
    fell_back = 0
    try:
        oflow = motion.get_method("LK")
        v = oflow(x[-3:])
        extrap = nowcasts.get_method("extrapolation")
        y = extrap(x[-1], v, OUT_LEN)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        # A silent fallback here is indistinguishable from persistence in the
        # output and would be reported as if it were pysteps. Count it.
        fell_back = 1
        _forecast.last_error = repr(exc)
        y = np.repeat(x[-1][None], OUT_LEN, axis=0)
    return idx, np.clip(y, 0, 255).transpose(1, 2, 0).astype(np.float32), fell_back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=H5,
                    help="SEVIR split to advect; any file with IN_vil")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/"
                                 "sevir_pred_pysteps.h5"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    f = h5py.File(P.require(args.h5, "SEVIR split"), "r")
    IN = f["IN_vil"]
    N = IN.shape[0] if not args.limit else min(args.limit, IN.shape[0])
    g = h5py.File(args.out, "w")
    d = g.create_dataset("pred_vil", shape=(N, 384, 384, OUT_LEN),
                         dtype="float32", chunks=(1, 384, 384, OUT_LEN),
                         compression="lzf")

    chunk = 128
    done = fallbacks = 0
    with mp.Pool(args.workers) as pool:
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            batch = [(i, IN[i]) for i in range(s, e)]
            for idx, y, fb in pool.imap_unordered(_forecast, batch):
                d[idx] = y
                done += 1
                fallbacks += fb
            print(f"  {done}/{N}  fallbacks={fallbacks}", flush=True)
    g.attrs["model"] = "pysteps_lk_extrapolation"
    g.attrs["n_fallback_to_persistence"] = fallbacks
    g.close()
    f.close()
    frac = fallbacks / max(N, 1)
    print(f"wrote {args.out}")
    print(f"fell back to persistence on {fallbacks}/{N} events ({frac:.1%})")
    if frac > 0.01:
        print("WARNING: fallback rate above 1%. These are NOT pysteps "
              "forecasts and must not be reported as such.")


if __name__ == "__main__":
    main()
