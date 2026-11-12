"""The exact control: fit the calibration map after pooling instead of before.

The paper fits one global quantile map per arm on raw pixels and applies it
before pooling (sevir/pooling_control.py). That map commutes with max
pooling and no pooling, so there it is exactly a per-arm threshold. Under
average pooling it need not commute, and a nonlinear map can reorder blocks.

Here the map is fitted on the pooled fields of the calibration half and applied
to the pooled forecast. A monotone map applied after pooling cannot reorder
pooled blocks under any pooling, so it is exactly a per-arm threshold on block
values. Two versions are reported:

  post_grid   the paper's convention (1001 percentiles, linear interpolation),
              fitted on pooled values of every calibration event;
  post_exact  the exact empirical quantile map: at each threshold, the forecast
              threshold whose calibration-half exceedance count equals the
              observed one. It is one monotone map evaluated at six points.

For 'none' pooling both equal the paper's map, which is reused.

The paper's own control (pre-pooling map, 'pre') and the raw score are
recomputed in the same pass and checked against the released
results/sevir_poolingq_<arm>.json.

Reads the cached prediction HDF5s and the SEVIR test HDF5. CPU only, numpy and
h5py. No model is run.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402
THRESHOLDS = [16, 74, 133, 160, 181, 219]
POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]
POOLED = POOLINGS[1:]
GT_BLOCK = 117          # chunk length of OUT_vil along the event axis
SUB = 13                # events per working batch (117 = 9 x 13)


def pool(x, mode):
    """x: (B, H, W, T) array. Returns pooled (B, H/k, W/k, T)."""
    if mode == "none":
        return x
    k = 4 if mode.endswith("4") else 16
    B, H, W, T = x.shape
    y = x.reshape(B, H // k, k, W // k, k, T)
    if mode.startswith("avg"):
        return y.mean(axis=(2, 4), dtype=np.float32)
    return y.max(axis=(2, 4))


def fit_grid(p, g, n_q=1001):
    qs = np.linspace(0, 100, n_q)
    src = np.maximum.accumulate(np.percentile(p, qs)).astype(np.float32)
    dst = np.maximum.accumulate(np.percentile(g, qs)).astype(np.float32)
    return src, dst


def apply_map(x, src, dst):
    """Same arithmetic as pooling_control._qmap (float32, clamped index,
    linear extrapolation beyond the end knots)."""
    flat = x.reshape(-1).astype(np.float32, copy=False)
    i = np.clip(np.searchsorted(src, flat, side="left"), 1, len(src) - 1)
    x0, x1, y0, y1 = src[i - 1], src[i], dst[i - 1], dst[i]
    d = x1 - x0
    w = np.where(d > 0, (flat - x0) / (d + np.float32(1e-12)), np.float32(0))
    return (y0 + w.astype(np.float32) * (y1 - y0)).reshape(x.shape)


def fit_pre(pred, gt, cal, sub=60_000_000):
    """pooling_control.fit_global_qmap, without torch. Deterministic: with the
    SEVIR shapes it takes 33 calibration events and no random subsample."""
    per = max(1, sub // (384 * 384 * 12))
    take = cal[:: max(1, len(cal) // per)][:per] if per < len(cal) else cal
    p = np.concatenate([pred[i].reshape(-1) for i in take])
    g = np.concatenate([gt[i].reshape(-1).astype(np.float32) for i in take])
    assert p.size <= sub, "subsample branch would need the seeded RNG"
    return fit_grid(p, g)


def counts(pp, gp, t):
    ph, gh = pp >= t, gp >= t
    h = int(np.count_nonzero(ph & gh))
    f = int(np.count_nonzero(ph & ~gh))
    m = int(np.count_nonzero(~ph & gh))
    return h, f, m


def csi_bias(h, f, m):
    return h / max(h + f + m, 1), (h + f) / max(h + m, 1)


def batches(n):
    for b0 in range(0, n, GT_BLOCK):
        b1 = min(n, b0 + GT_BLOCK)
        yield b0, b1


def run(arms, gt_path, out_dir):
    fg = h5py.File(gt_path, "r")
    G = fg["OUT_vil"]
    N = G.shape[0]
    cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)
    fps = {a: h5py.File(p, "r") for a, p in arms.items()}
    P = {a: f["pred_vil"] for a, f in fps.items()}
    for a in arms:
        assert P[a].shape == G.shape, (a, P[a].shape, G.shape)

    t0 = time.time()
    pre = {a: fit_pre(P[a], G, cal) for a in arms}
    print(f"pre-pooling maps fitted ({time.time() - t0:.0f}s)", flush=True)

    # Pass 1: pooled values on the calibration half.
    store = {("gt", m): [] for m in POOLED}
    store.update({(a, m): [] for a in arms for m in POOLED})
    for b0, b1 in batches(N):
        gblk = G[b0:b1]
        for s0 in range(b0, b1, SUB):
            s1 = min(b1, s0 + SUB)
            ids = np.arange(s0, s1)
            keep = ids % 2 == 0
            if not keep.any():
                continue
            g = gblk[s0 - b0:s1 - b0][keep]
            for m in POOLED:
                store[("gt", m)].append(pool(g, m).reshape(-1))
            for a in arms:
                p = P[a][s0:s1][keep]
                for m in POOLED:
                    store[(a, m)].append(pool(p, m).reshape(-1))
        print(f"  pass 1: {b1}/{N} events ({time.time() - t0:.0f}s)", flush=True)
    for k in list(store):
        store[k] = np.concatenate(store.pop(k))

    post = {}
    for a in arms:
        for m in POOLED:
            pc, gc = store[(a, m)], store[("gt", m)]
            src, dst = fit_grid(pc, gc)
            srt = np.sort(pc)[::-1]
            exact = {}
            for t in THRESHOLDS:
                k = int(np.count_nonzero(gc >= t))
                exact[t] = float(srt[k - 1]) if k > 0 else float("inf")
            post[(a, m)] = (src, dst, exact)
    del store
    print(f"post-pooling maps fitted ({time.time() - t0:.0f}s)", flush=True)

    # Pass 2: contingency counts on the evaluation half.
    kinds = ("raw", "pre", "post_grid", "post_exact")
    acc = {(a, m, k, t): [0, 0, 0] for a in arms for m in POOLINGS
           for k in kinds for t in THRESHOLDS}
    for b0, b1 in batches(N):
        gblk = G[b0:b1]
        for s0 in range(b0, b1, SUB):
            s1 = min(b1, s0 + SUB)
            ids = np.arange(s0, s1)
            keep = ids % 2 == 1
            if not keep.any():
                continue
            g = gblk[s0 - b0:s1 - b0][keep].astype(np.float32)
            gp = {m: pool(g, m) for m in POOLINGS}
            for a in arms:
                p = P[a][s0:s1][keep]
                q = apply_map(p, *pre[a])
                for m in POOLINGS:
                    pr, pq = pool(p, m), pool(q, m)
                    if m == "none":
                        pg, thr_ex = pq, None
                    else:
                        src, dst, exact = post[(a, m)]
                        pg, thr_ex = apply_map(pr, src, dst), exact
                    for t in THRESHOLDS:
                        for k, arr in (("raw", pr), ("pre", pq), ("post_grid", pg)):
                            c = counts(arr, gp[m], t)
                            cell = acc[(a, m, k, t)]
                            for j in range(3):
                                cell[j] += c[j]
                        if thr_ex is None:
                            c = counts(pq, gp[m], t)
                        else:
                            ph, gh = pr >= thr_ex[t], gp[m] >= t
                            c = (int(np.count_nonzero(ph & gh)),
                                 int(np.count_nonzero(ph & ~gh)),
                                 int(np.count_nonzero(~ph & gh)))
                        cell = acc[(a, m, "post_exact", t)]
                        for j in range(3):
                            cell[j] += c[j]
        print(f"  pass 2: {b1}/{N} events ({time.time() - t0:.0f}s)", flush=True)

    for f in fps.values():
        f.close()
    fg.close()

    for a in arms:
        gold = json.loads((P.RESULTS /
                           f"sevir_poolingq_{a}.json").read_text())
        gold = {(r["pooling"], r["threshold"]): r for r in gold["table"]}
        table, worst = [], 0.0
        for m in POOLINGS:
            for t in THRESHOLDS:
                row = {"pooling": m, "threshold": t}
                for k in kinds:
                    c, b = csi_bias(*acc[(a, m, k, t)])
                    row[f"csi_{k}"], row[f"bias_{k}"] = c, b
                    row[f"counts_{k}"] = acc[(a, m, k, t)]
                if m != "none":
                    row["post_exact_threshold"] = post[(a, m)][2][t]
                gr = gold[(m, t)]
                row["check_raw"] = row["csi_raw"] - gr["csi_uncal"]
                row["check_pre"] = row["csi_pre"] - gr["csi_qmap_global"]
                worst = max(worst, abs(row["check_raw"]), abs(row["check_pre"]))
                table.append(row)
        out = {"arm": a, "pred": arms[a].name, "n_calib": len(cal),
               "n_eval": len(ev), "max_abs_check": worst, "table": table}
        fn = out_dir / f"sevir_postpool_{a}.json"
        fn.write_text(json.dumps(out, indent=2))
        print(f"{a}: wrote {fn.name}; max |CSI - released| = {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path/to/sevir_pred.h5; name must match "
                         "results/sevir_poolingq_<name>.json")
    ap.add_argument("--gt", type=Path,
                    default=P.SEVIR_H5)
    ap.add_argument("--out-dir", type=Path, default=P.RESULTS / "revision" / "postpool")
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arm)
    run({k: Path(v) for k, v in arms.items()}, args.gt, args.out_dir)


if __name__ == "__main__":
    main()
