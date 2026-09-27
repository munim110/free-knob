"""Event-level cluster bootstrap for the pairwise-contrast claims.

published_contrasts.py reports a correlation over 450 cells and a count of sign
reversals among them. Those cells are not independent: 15 pairs are formed from 6
arms, every pair is evaluated at 30 (pooling, threshold) cells, and every cell is
a ratio over the same 936 evaluation events. A correlation computed over them
carries no interval, and neither does the reversal count.

The resampling unit here is the event. Per-event contingency counts are cached
once for each arm under the identity knob and under that arm's global quantile
map, then events are resampled with replacement. Inside each replicate every
pooled CSI, every pooled bias, all 450 contrasts and the whole correlation are
recomputed from the resampled counts, so the interval propagates the shared-event
dependence rather than assuming it away.

One event-index vector is shared by all arms within a replicate, which keeps the
contrasts paired.

Two summaries are reported. The cell-level one treats all 450 cells as the
sample, which is what the manuscript quotes. The pair-level one first averages
within each of the 15 pairs and correlates the 15 pair means, which is the
conservative reading if one regards the pair rather than the cell as the unit.
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

import sys as _s2
from pathlib import Path as _P2
_s2.path.insert(0, str(_P2(__file__).resolve().parents[1] / "tools"))
_s2.path.insert(0, str(_P2(__file__).resolve().parent))
import paths as P                                          # noqa: E402
from pooling_control import (POOLINGS, THRESHOLDS,          # noqa: E402
                             fit_global_qmap, pool)

ARMS = ["persistence", "pysteps", "earthformer", "cascast_det",
        "cascast_cascade", "cascast_cascade_cfg1"]
KNOBS = ["identity", "qmap_global"]

# The knob fitted on the prior-period window of Appendix J. Caching it in the
# SAME pass as the other two is the whole point: reading and pooling the
# prediction fields is what costs half an hour, and it is identical work for
# every knob. Fitting the map is seconds and the resampling afterwards is
# numpy, so a cache that carries this column lets the val-fit intervals be
# computed on any laptop, with no HDF5 and no second pass.
VAL_KNOB = "qmap_val"


def per_event_counts(pred_h5, gt_h5, idx, knobs, device, batch=16):
    """counts[e, pooling, threshold, knob] = (hits, false alarms, misses).

    `knobs` is an ordered mapping name -> callable, applied to the raw field.
    """
    names = list(knobs)
    out = np.zeros((len(idx), len(POOLINGS), len(THRESHOLDS), len(names), 3),
                   dtype=np.int64)
    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    Pv, Gv = fp["pred_vil"], fg["OUT_vil"]
    for s in range(0, len(idx), batch):
        sel = idx[s:s + batch]
        order = np.argsort(sel)
        srt = sel[order]
        p = torch.from_numpy(Pv[srt].astype(np.float32)).to(device)
        g = torch.from_numpy(Gv[srt].astype(np.float32)).to(device)
        p, g = p.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        inv = np.argsort(order)
        for ki, kn in enumerate(names):
            pk = knobs[kn](p)
            for pi, pl in enumerate(POOLINGS):
                pp, gp = pool(pk, pl), pool(g, pl)
                for ti, t in enumerate(THRESHOLDS):
                    ph, gh = pp >= t, gp >= t
                    dims = (1, 2, 3)
                    h = torch.count_nonzero(ph & gh, dim=dims)
                    f = torch.count_nonzero(ph & ~gh, dim=dims)
                    m = torch.count_nonzero(~ph & gh, dim=dims)
                    trip = torch.stack([h, f, m], -1).cpu().numpy()
                    out[s:s + len(sel), pi, ti, ki] = trip[inv]
    fp.close(); fg.close()
    return out


def build_qmap(pred_h5, gt_h5, cal, device):
    src, dst = fit_global_qmap(pred_h5, gt_h5, cal)
    s = torch.from_numpy(src).to(device)
    t = torch.from_numpy(dst).to(device)

    def _q(x):
        flat = x.reshape(-1)
        i = torch.searchsorted(s, flat).clamp(1, len(s) - 1)
        x0, x1, y0, y1 = s[i - 1], s[i], t[i - 1], t[i]
        w = torch.where(x1 > x0, (flat - x0) / (x1 - x0 + 1e-12),
                        torch.zeros_like(flat))
        return (y0 + w * (y1 - y0)).reshape(x.shape)
    return _q


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / d if d > 0 else float("nan")


def replicate_stats(tot, pairs, masks):
    """tot[arm, pooling, threshold, knob] = (h, f, m) summed over a resample."""
    h, f, m = tot[..., 0], tot[..., 1], tot[..., 2]
    csi = h / np.maximum(h + f + m, 1)
    bias = (h + f) / np.maximum(h + m, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(np.log(np.where(bias > 0, bias, np.nan)))
    devgap, distort, raw, cal, pair_id = [], [], [], [], []
    for pi, (a, b) in enumerate(pairs):
        da, db = dev[a, :, :, 0], dev[b, :, :, 0]
        r = csi[b, :, :, 0] - csi[a, :, :, 0]
        c = csi[b, :, :, 1] - csi[a, :, :, 1]
        devgap.append((da - db).ravel())
        distort.append((r - c).ravel())
        raw.append(r.ravel())
        cal.append(c.ravel())
        pair_id.append(np.full(r.size, pi))
    devgap = np.concatenate(devgap); distort = np.concatenate(distort)
    raw = np.concatenate(raw); cal = np.concatenate(cal)
    pair_id = np.concatenate(pair_id)
    ok = np.isfinite(devgap) & np.isfinite(distort)

    out = {}
    for tag, sel in masks.items():
        s = sel & ok
        out[f"r_{tag}"] = pearson(devgap[s], distort[s])
    flip = (raw > 0) != (cal > 0)
    out["flips"] = int(np.count_nonzero(flip & ok))
    out["flips_big"] = int(np.count_nonzero(
        flip & ok & (np.minimum(np.abs(raw), np.abs(cal)) > 0.01)))
    # pair-level: average within each pair, then correlate the 15 means
    pm_x, pm_y = [], []
    for pi in range(len(pairs)):
        s = (pair_id == pi) & ok
        if s.any():
            pm_x.append(devgap[s].mean())
            pm_y.append(distort[s].mean())
    out["r_pair"] = pearson(pm_x, pm_y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=P.RESULTS)
    ap.add_argument("--gt", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_perevent_counts.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_contrast_bootstrap.json"))
    # the prior-period window of Appendix J; when present its map is cached as a
    # third knob in the same pass over the fields
    ap.add_argument("--val-gt", type=Path,
                    default=P.SEVIR_DATA / "nowcast_val_subset.h5")
    ap.add_argument("--val-prefix", default="sevir_valpred_")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = [a for a in ARMS
            if (args.results / f"sevir_pred_{a}.h5").exists() or a == "persistence"]

    if args.cache.exists():
        z = np.load(args.cache)
        counts = z["counts"]
        arms = list(z["arms"])
        ev = z["ev"]
        # caches written before the val-fit column existed carry no knob names
        knobs_cached = ([str(k) for k in z["knobs"]] if "knobs" in z.files
                        else KNOBS[:counts.shape[4]])
        print(f"loaded cached counts {counts.shape} from {args.cache} "
              f"(knobs: {', '.join(knobs_cached)})")
        if VAL_KNOB not in knobs_cached:
            print(f"  note: no {VAL_KNOB} column. Delete the cache and rerun "
                  f"with the sevir_valpred_*.h5 caches present to add it, "
                  f"which also lets sevir/valfit_bootstrap.py run.")
    else:
        with h5py.File(args.gt, "r") as f:
            N = f["OUT_vil"].shape[0]
        cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)
        val_gt = args.val_gt if args.val_gt.exists() else None
        if val_gt is None:
            print(f"note: {args.val_gt} absent, caching {len(KNOBS)} knobs only; "
                  f"the val-fit intervals of Appendix J will need a second pass")
        blocks, knobs_cached = [], None
        for a in arms:
            ph = args.results / f"sevir_pred_{a}.h5"
            print(f"{a}: fitting qmap on {len(cal)} calibration events")
            ks = {"identity": lambda x: x,
                  "qmap_global": build_qmap(ph, args.gt, cal, device)}
            vph = args.results / f"{args.val_prefix}{a}.h5"
            if val_gt is not None and vph.exists():
                with h5py.File(val_gt, "r") as f:
                    vidx = np.arange(f["OUT_vil"].shape[0])
                ks[VAL_KNOB] = build_qmap(vph, val_gt, vidx, device)
            elif val_gt is not None:
                print(f"  {a}: no {vph.name}, skipping the {VAL_KNOB} column")
            if knobs_cached is None:
                knobs_cached = list(ks)
            elif list(ks) != knobs_cached:
                raise SystemExit(
                    f"arm {a} has knobs {list(ks)} but earlier arms have "
                    f"{knobs_cached}; the cache must be rectangular. Generate "
                    f"the missing sevir_valpred_*.h5 or remove the val subset.")
            print(f"{a}: accumulating per-event counts on {len(ev)} events "
                  f"under {len(ks)} knobs")
            blocks.append(per_event_counts(ph, args.gt, ev, ks, device))
        counts = np.stack(blocks)                    # (arm, ev, pool, thr, knob, 3)
        counts = counts.transpose(1, 0, 2, 3, 4, 5)  # (ev, arm, ...)
        np.savez_compressed(args.cache, counts=counts, arms=np.array(arms),
                            ev=ev, knobs=np.array(knobs_cached))
        print(f"cached {counts.shape} to {args.cache} "
              f"(knobs: {', '.join(knobs_cached)})")

    pairs = [(i, j) for i in range(len(arms)) for j in range(i + 1, len(arms))]
    npl, nth = len(POOLINGS), len(THRESHOLDS)
    grid_pl = np.repeat(np.arange(npl), nth)
    grid_th = np.tile(np.array(THRESHOLDS), npl)
    cell = np.tile(np.stack([grid_pl, grid_th]), (1, len(pairs)))
    masks = {
        "all": np.ones(cell.shape[1], bool),
        "max16": cell[0] == POOLINGS.index("max16"),
        "extreme": cell[1] >= 181,
    }

    point = replicate_stats(counts.sum(0), pairs, masks)
    print("point estimates:", {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in point.items()})

    rng = np.random.default_rng(args.seed)
    n_ev = counts.shape[0]
    reps = {k: [] for k in point}
    for b in range(args.n_boot):
        pick = rng.integers(0, n_ev, n_ev)
        s = replicate_stats(counts[pick].sum(0), pairs, masks)
        for k, v in s.items():
            reps[k].append(v)
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{args.n_boot}")

    out = {"n_boot": args.n_boot, "seed": args.seed, "arms": list(arms),
           "n_events": int(n_ev), "n_pairs": len(pairs),
           "n_cells": int(cell.shape[1]), "point": point, "ci": {}}
    for k, v in reps.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        out["ci"][k] = {"lo": float(np.percentile(v, 2.5)),
                        "hi": float(np.percentile(v, 97.5)),
                        "mean": float(v.mean())}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n{'quantity':>12} {'point':>9} {'95% interval':>22}")
    for k in point:
        c = out["ci"][k]
        p = point[k]
        fmt = "{:9.3f}" if isinstance(p, float) else "{:9d}"
        print(f"{k:>12} " + fmt.format(p) +
              f"   [{c['lo']:.3f}, {c['hi']:.3f}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
