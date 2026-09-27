"""Reported skill and calibration gain as a function of the pooling convention.

CasCast reports csi, csi-4-avg, csi-16-avg, csi-4-max, csi-16-max and quotes its
headline extreme-threshold gains at POOL16. Pooling before thresholding is a
drastic transformation that interacts with how sharp a model is: max-pooling
forgives displacement and rewards any high value in a block, average-pooling
suppresses isolated peaks. Both change frequency bias, so both change how much a
free monotone recalibration can buy.

For every (pooling, threshold) we report the model's own skill and the skill
after a knob chosen on a disjoint half by pooled bias -> 1.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

THRESHOLDS = [16, 74, 133, 160, 181, 219]
POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]
SHIFTS = [-8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32]
MULS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]


def pool(x, mode):
    """x: torch (B,T,H,W) on device."""
    if mode == "none":
        return x
    k = 4 if mode.endswith("4") else 16
    B, T, H, W = x.shape
    y = x.reshape(B * T, 1, H, W)
    y = (F.avg_pool2d(y, k) if mode.startswith("avg")
         else F.max_pool2d(y, k))
    return y.reshape(B, T, H // k, W // k)


def counts(pred, gt, thr):
    ph, gh = pred >= thr, gt >= thr
    return (torch.count_nonzero(ph & gh).item(),
            torch.count_nonzero(ph & ~gh).item(),
            torch.count_nonzero(~ph & gh).item())


def csi_bias(h, f, m):
    return (h / max(h + f + m, 1), (h + f) / max(h + m, 1))


def fit_global_qmap(pred_h5, gt_h5, idx, n_q=1001, sub=60_000_000, seed=0):
    """ONE monotone function, fitted on held-out events, applied to the raw
    field before any thresholding or pooling.

    This is the answer to "your knob is chosen per threshold": a quantile map is
    a single transform, identical at every threshold and every pooling scale.
    """
    rng = np.random.default_rng(seed)
    ps, gs = [], []
    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    per = max(1, sub // (384 * 384 * 12))
    take = idx[:: max(1, len(idx) // max(per, 1))][:per] if per < len(idx) else idx
    for i in take:
        ps.append(fp["pred_vil"][i].reshape(-1))
        gs.append(fg["OUT_vil"][i].reshape(-1).astype(np.float32))
    fp.close(); fg.close()
    p = np.concatenate(ps); g = np.concatenate(gs)
    if p.size > sub:
        sel = rng.choice(p.size, size=sub, replace=False)
        p, g = p[sel], g[sel]
    qs = np.linspace(0, 100, n_q)
    src = np.maximum.accumulate(np.percentile(p, qs))
    dst = np.maximum.accumulate(np.percentile(g, qs))
    return src.astype(np.float32), dst.astype(np.float32)


def accumulate(pred_h5, gt_h5, idx, knobs, device, batch=32):
    """Return acc[pooling][knob][thr] = [hits, fas, misses]."""
    acc = {p: {k: {t: [0, 0, 0] for t in THRESHOLDS} for k in knobs}
           for p in POOLINGS}
    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    P, G = fp["pred_vil"], fg["OUT_vil"]
    for s in range(0, len(idx), batch):
        sel = np.sort(idx[s:s + batch])
        p = torch.from_numpy(P[sel].astype(np.float32)).to(device)
        g = torch.from_numpy(G[sel].astype(np.float32)).to(device)
        p, g = p.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)   # (B,T,H,W)
        for pl in POOLINGS:
            gp = pool(g, pl)
            for kname, fn in knobs.items():
                pp = pool(fn(p), pl)
                for t in THRESHOLDS:
                    h, f, m = counts(pp, gp, t)
                    a = acc[pl][kname][t]
                    a[0] += h; a[1] += f; a[2] += m
    fp.close(); fg.close()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path,
                    default=P.SEVIR_H5)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knobs = {"identity": lambda x: x}
    for d in SHIFTS:
        knobs[f"shift={d}"] = (lambda d: lambda x: x + d)(d)
    for g in MULS:
        knobs[f"mul={g}"] = (lambda g: lambda x: x * g)(g)

    with h5py.File(args.pred, "r") as f:
        N = f["pred_vil"].shape[0]
    cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)

    # a single global monotone transform, fitted once on the calibration half
    src, dst = fit_global_qmap(args.pred, args.gt, cal)
    src_t = torch.from_numpy(src).to(device)
    dst_t = torch.from_numpy(dst).to(device)

    def _qmap(x):
        flat = x.reshape(-1)
        i = torch.searchsorted(src_t, flat).clamp(1, len(src_t) - 1)
        x0, x1 = src_t[i - 1], src_t[i]
        y0, y1 = dst_t[i - 1], dst_t[i]
        w = torch.where(x1 > x0, (flat - x0) / (x1 - x0 + 1e-12),
                        torch.zeros_like(flat))
        return (y0 + w * (y1 - y0)).reshape(x.shape)

    knobs["qmap_global"] = _qmap
    print("  fitted one global quantile map on the calibration half")
    print(f"{args.name}: N={N} calib={len(cal)} eval={len(ev)} "
          f"knobs={len(knobs)} poolings={len(POOLINGS)}")

    A_cal = accumulate(args.pred, args.gt, cal, knobs, device)
    print("  calib accumulated")
    A_ev = accumulate(args.pred, args.gt, ev, knobs, device)
    print("  eval accumulated")

    out = {"name": args.name, "thresholds": THRESHOLDS, "poolings": POOLINGS,
           "n_calib": len(cal), "n_eval": len(ev), "table": []}
    for pl in POOLINGS:
        for t in THRESHOLDS:
            base_c, base_b = csi_bias(*A_ev[pl]["identity"][t])
            # pick knob on the calibration half by |bias - 1|
            best_k, best_dev = None, None
            for k in knobs:
                _, b = csi_bias(*A_cal[pl][k][t])
                dev = abs(b - 1.0)
                if best_dev is None or dev < best_dev:
                    best_dev, best_k = dev, k
            sel_c, sel_b = csi_bias(*A_ev[pl][best_k][t])
            oracle = max(csi_bias(*A_ev[pl][k][t])[0] for k in knobs)
            # the single global transform, identical at every cell
            gq_c, gq_b = csi_bias(*A_ev[pl]["qmap_global"][t])
            out["table"].append(
                {"pooling": pl, "threshold": t,
                 "csi_uncal": base_c, "bias_uncal": base_b,
                 "csi_recal": sel_c, "bias_recal": sel_b,
                 "knob": best_k, "csi_oracle": oracle,
                 "csi_qmap_global": gq_c, "bias_qmap_global": gq_b,
                 "rel_gain_qmap_global": ((gq_c - base_c) / base_c
                                          if base_c > 0 else None),
                 "rel_gain": (sel_c - base_c) / base_c if base_c > 0 else None})
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n{'pool':>6} {'thr':>5} {'CSI unc':>8} {'bias':>7} "
          f"{'CSI recal':>10} {'rel':>8} | {'CSI qmapG':>10} {'relG':>8}  knob")
    for r in out["table"]:
        rg = f"{r['rel_gain']:+.1%}" if r["rel_gain"] is not None else "n/a"
        rgg = (f"{r['rel_gain_qmap_global']:+.1%}"
               if r["rel_gain_qmap_global"] is not None else "n/a")
        print(f"{r['pooling']:>6} {r['threshold']:>5} {r['csi_uncal']:8.4f} "
              f"{r['bias_uncal']:7.3f} {r['csi_recal']:10.4f} {rg:>8} | "
              f"{r['csi_qmap_global']:10.4f} {rgg:>8}  {r['knob']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
