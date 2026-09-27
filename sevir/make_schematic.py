"""Cache one event's fields for the mechanism panel of Figure 1.

The mechanism is stated in prose in Section 4 and measured in the two right-hand
panels of Figure 1, but neither shows what max-pooling does to a sharp field and
a smooth one. This caches the arrays for that picture: one observed frame, the
same frame as a mean-error-trained model predicts it, and the two
sixteen-by-sixteen max-pooled masks at the headline threshold.

The frame is chosen by a rule rather than by eye, and the rule is written to be
neither flattering nor damning. Picking the frame with the largest gap would
overstate the mechanism and picking the busiest frame understates it, because the
model does relatively better where there is more to hit. So among evaluation
frames with enough observed blocks to see, we take the one whose ratio of
forecast to observed positive blocks is closest to the arm's aggregate pooled
frequency bias over the whole split. The picture then shows a typical frame by
the very statistic the paper reports.

The cache is a few hundred kilobytes, which keeps Figure 1 reproducible from the
released artefacts alone. Everything upstream of it needs the prediction HDF5s.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

import sys as _s2
from pathlib import Path as _P2
_s2.path.insert(0, str(_P2(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402


def max_pool(x, k):
    h, w = x.shape
    return x[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).max((1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--pred", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_pred_cascast_det.h5"))
    ap.add_argument("--arm", default="CasCast backbone")
    ap.add_argument("--threshold", type=int, default=219)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--search", type=int, default=200,
                    help="evaluation events to scan")
    ap.add_argument("--min-blocks", type=int, default=24,
                    help="observed positive blocks needed for the panel to read")
    ap.add_argument("--pooling-json", type=Path,
                    default=Path(str(P.RESULTS)
                                 + "/sevir_poolingq_cascast_det.json"),
                    help="supplies the aggregate pooled bias the frame is "
                         "matched against")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_schematic.npz"))
    args = ap.parse_args()

    P.require(args.gt, "SEVIR test split")
    P.require(args.pred, "cached prediction")
    P.require(args.pooling_json, "pooling sweep for this arm")

    import json
    tbl = json.loads(args.pooling_json.read_text())["table"]
    target = next(r["bias_uncal"] for r in tbl
                  if r["pooling"] == f"max{args.pool}"
                  and r["threshold"] == args.threshold)

    k, thr = args.pool, args.threshold
    best, bi, bt, bratio = None, None, None, None
    with h5py.File(args.gt, "r") as fg, h5py.File(args.pred, "r") as fp:
        G, Pd = fg["OUT_vil"], fp["pred_vil"]
        for i in np.arange(1, G.shape[0], 2)[:args.search]:
            g, p = G[i].astype(np.float32), Pd[i].astype(np.float32)
            for t in range(g.shape[-1]):
                gm = max_pool(g[..., t], k) >= thr
                if gm.sum() < args.min_blocks:
                    continue
                pm = max_pool(p[..., t], k) >= thr
                ratio = pm.sum() / gm.sum()
                d = abs(ratio - target)
                if best is None or d < best:
                    best, bi, bt, bratio = d, int(i), t, ratio
        if bi is None:
            raise SystemExit(f"no frame with >= {args.min_blocks} observed "
                             f"blocks in the first {args.search} events")
        gt = G[bi][..., bt].astype(np.float32)
        pred = Pd[bi][..., bt].astype(np.float32)

    gm, pm = max_pool(gt, k) >= thr, max_pool(pred, k) >= thr
    np.savez_compressed(
        args.out, gt=gt.astype(np.uint8),
        pred=np.clip(pred, 0, 255).astype(np.uint8),
        gt_mask=gm, pred_mask=pm, event=bi, frame=bt,
        threshold=thr, pool=k, arm=args.arm,
        frame_bias=bratio, split_bias=target)
    print(f"aggregate pooled bias for this arm at max{k}, tau={thr}: "
          f"{target:.3f}")
    print(f"event {bi} frame {bt}: block ratio {bratio:.3f}, "
          f"observed {gm.sum()} blocks, forecast {pm.sum()} of {gm.size}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
