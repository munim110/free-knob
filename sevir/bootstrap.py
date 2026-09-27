"""Event-level paired bootstrap for the SEVIR comparisons.

Every number in the pooling/decomposition tables is a single pooled contingency
ratio over ~936 events, so it carries no uncertainty on its face. This script
caches per-event (hits, false alarms, misses) for each arm and then resamples
events with replacement, recomputing the pooled CSI inside each replicate. The
resampling is *paired*: one event index vector is shared by all arms in a
replicate, so the CI on a difference reflects the between-arm correlation
rather than treating the arms as independent samples.

Arms are named `<model>:<knob>`, where knob is one of
    identity          the model as released
    qmap_global       the single global monotone quantile map, fitted on the
                      disjoint calibration half and never on the eval half
    qmap_val          the same map fitted instead on the prior-period validation
                      window of Appendix J, which both decomposed models held out
                      of training and which ends before the test period starts.
                      Needs the `sevir_valpred_<arm>.h5` caches and the val
                      subset; use it to put an interval on the residual under the
                      fit a deployed system would actually have
    shift=d / mul=g   the per-cell knob selected on the calibration half

Only the evaluation half is bootstrapped; the calibration half fixes the knob
once and is not resampled, so knob selection cannot leak into the CI.
"""
import argparse
import json
from pathlib import Path

import sys
from pathlib import Path as _Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as P                                    # noqa: E402
from pooling_control import (POOLINGS, THRESHOLDS,   # noqa: E402
                             fit_global_qmap, pool)



def make_knob(spec, device, pred_h5, gt_h5, cal_idx, val=None):
    if spec == "identity":
        return lambda x: x
    if spec.startswith("shift="):
        d = float(spec.split("=")[1])
        return lambda x: x + d
    if spec.startswith("mul="):
        g = float(spec.split("=")[1])
        return lambda x: x * g
    if spec in ("qmap_global", "qmap_val"):
        if spec == "qmap_global":
            fit_pred, fit_gt, fit_idx = pred_h5, gt_h5, cal_idx
        else:
            # The same single global map, fitted on the prior-period validation
            # window of Appendix J instead of on the test calibration half, so
            # the interval on the residual can be quoted under the fit a
            # deployed system would actually have. `val` is (val_pred, val_gt),
            # resolved in main() from the arm's own prediction filename.
            fit_pred, fit_gt = val
            P.require(fit_pred, "validation prediction cache")
            P.require(fit_gt, "validation subset (run sevir/make_val_subset.py)")
            with h5py.File(fit_gt, "r") as f:
                fit_idx = np.arange(f["OUT_vil"].shape[0])
        src, dst = fit_global_qmap(fit_pred, fit_gt, fit_idx)
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
    raise ValueError(f"unknown knob {spec!r}")


def per_event_counts(pred_h5, gt_h5, idx, knob_fn, device, batch=16):
    """(n_events, n_pool, n_thr, 3) int64 of hits / false alarms / misses."""
    out = np.zeros((len(idx), len(POOLINGS), len(THRESHOLDS), 3), np.int64)
    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    P, G = fp["pred_vil"], fg["OUT_vil"]
    for s in range(0, len(idx), batch):
        sel = np.sort(idx[s:s + batch])
        p = torch.from_numpy(P[sel].astype(np.float32)).to(device).permute(0, 3, 1, 2)
        g = torch.from_numpy(G[sel].astype(np.float32)).to(device).permute(0, 3, 1, 2)
        p = knob_fn(p)
        for pi, pl in enumerate(POOLINGS):
            pp, gp = pool(p, pl), pool(g, pl)
            for ti, t in enumerate(THRESHOLDS):
                ph, gh = pp >= t, gp >= t
                # sum over (T,H,W) -> one triple per event
                out[s:s + len(sel), pi, ti, 0] = (ph & gh).sum((1, 2, 3)).cpu()
                out[s:s + len(sel), pi, ti, 1] = (ph & ~gh).sum((1, 2, 3)).cpu()
                out[s:s + len(sel), pi, ti, 2] = (~ph & gh).sum((1, 2, 3)).cpu()
    fp.close(); fg.close()
    return out


def csi_from(c):
    """c: (..., 3) summed counts -> pooled CSI."""
    h, f, m = c[..., 0], c[..., 1], c[..., 2]
    return h / np.maximum(h + f + m, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    metavar="LABEL=PRED_H5:KNOB",
                    help="repeatable, e.g. "
                         "'det+cal=results/sevir_pred_cascast_det.h5:qmap_global'")
    ap.add_argument("--gt", type=Path,

                    default=P.SEVIR_H5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    # only needed by the qmap_val knob, which fits the map on the prior-period
    # window of Appendix J rather than on the test calibration half
    ap.add_argument("--val-gt", type=Path,
                    default=P.SEVIR_DATA / "nowcast_val_subset.h5")
    ap.add_argument("--val-prefix", default="sevir_valpred_")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts, specs = {}, {}
    n_ev = None
    for a in args.arm:
        label, rhs = a.split("=", 1)
        pred, knob = rhs.rsplit(":", 1)
        pred = Path(pred)
        with h5py.File(pred, "r") as f:
            N = f["pred_vil"].shape[0]
        cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)
        # sevir_pred_<arm>.h5 -> <val-prefix><arm>.h5 in the same directory,
        # which is the naming val_fitted_control.py already writes
        val = (pred.with_name(args.val_prefix
                              + pred.stem.replace("sevir_pred_", "") + ".h5"),
               args.val_gt)
        fn = make_knob(knob, device, pred, args.gt, cal, val)
        counts[label] = per_event_counts(pred, args.gt, ev, fn, device)
        specs[label] = {"pred": str(pred), "knob": knob, "n_eval": len(ev)}
        n_ev = len(ev)
        print(f"  cached {label}  ({pred.name}, knob={knob})", flush=True)

    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0, n_ev, size=(args.n_boot, n_ev))   # shared -> paired

    labels = list(counts)
    point = {L: csi_from(counts[L].sum(0)) for L in labels}
    reps = {}
    for L in labels:
        c = counts[L]
        # (n_boot, n_pool, n_thr, 3) by summing the resampled events
        reps[L] = np.stack([csi_from(c[b].sum(0)) for b in boot])

    rows = []
    for pi, pl in enumerate(POOLINGS):
        for ti, t in enumerate(THRESHOLDS):
            cell = {"pooling": pl, "threshold": t, "arms": {}, "contrasts": {}}
            for L in labels:
                d = reps[L][:, pi, ti]
                cell["arms"][L] = {
                    "csi": float(point[L][pi, ti]),
                    "ci95": [float(np.percentile(d, 2.5)),
                             float(np.percentile(d, 97.5))]}
            for i, A in enumerate(labels):
                for B in labels[i + 1:]:
                    d = reps[A][:, pi, ti] - reps[B][:, pi, ti]
                    obs = float(point[A][pi, ti] - point[B][pi, ti])
                    # two-sided bootstrap p for "difference is zero"
                    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                    cell["contrasts"][f"{A} - {B}"] = {
                        "diff": obs,
                        "ci95": [float(np.percentile(d, 2.5)),
                                 float(np.percentile(d, 97.5))],
                        "p": float(min(p, 1.0))}
            rows.append(cell)

    args.out.write_text(json.dumps(
        {"arms": specs, "n_boot": args.n_boot, "n_eval": n_ev,
         "table": rows}, indent=2))

    print(f"\n{'pool':>6} {'thr':>5} " +
          " ".join(f"{L:>22}" for L in labels))
    for r in rows:
        cells = " ".join(
            f"{r['arms'][L]['csi']:8.4f} [{r['arms'][L]['ci95'][0]:.3f},"
            f"{r['arms'][L]['ci95'][1]:.3f}]" for L in labels)
        print(f"{r['pooling']:>6} {r['threshold']:>5} {cells}")
    print("\ncontrasts at max16:")
    for r in rows:
        if r["pooling"] != "max16":
            continue
        for k, v in r["contrasts"].items():
            print(f"  thr={r['threshold']:>3} {k:>46}: {v['diff']:+.4f} "
                  f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
