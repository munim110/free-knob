"""Does a quantile map fitted per lead time erase more than one fitted globally?

Section 9 records the per-horizon map as "the natural next tightening" and
asserts it "can only lower the residual, never raise it". That is an in-sample
argument. The per-lead-time family strictly contains the global one -- setting
all twelve maps equal recovers it -- so on the calibration half it must match the
observed marginal at least as well. Out of sample it need not: each of the twelve
maps is fitted on a twelfth of the pixels, and the upper percentiles that decide
a tau=219 cell are estimated from a twelfth as many exceedances.

Two budgets separate the two explanations.

  matched   every map, global or per-lead, is fitted from the SAME sampled
            calibration events, so a per-lead map sees 1/12 the pixels. This is
            the honest like-for-like comparison and the one the paper would
            report: same data, larger family.

  equal     each per-lead map is given as many pixels as the global map by
            drawing more calibration events. If per-lead wins here and loses at
            the matched budget, the cost is estimation noise rather than the
            family.

Scoring is unchanged from pooling_control.py: pool, threshold, accumulate the
contingency table over events AND lead times, so the cell is the one the paper
quotes. A per-lead map can reorder a pixel at lead 3 against one at lead 9, but
never two pixels within a frame, and pooling is spatial while the tables are
summed across frames -- so no scored quantity sees the cross-frame reordering,
and the transform is exactly as free as the global one at these cells.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import sys as _s
from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
_s.path.insert(0, str(_P(__file__).resolve().parent))
import paths as P                                    # noqa: E402
from pooling_control import POOLINGS, THRESHOLDS, pool   # noqa: E402

N_Q = 1001
PIX_PER_LEAD = 384 * 384


def _percentile_map(p, g, n_q=N_Q):
    qs = np.linspace(0, 100, n_q)
    src = np.maximum.accumulate(np.percentile(p, qs))
    dst = np.maximum.accumulate(np.percentile(g, qs))
    return src.astype(np.float32), dst.astype(np.float32)


def _take(cal, n):
    """The spread subsample fit_global_qmap draws: a fixed stride over the whole
    calibration half, never a contiguous block."""
    n = min(n, len(cal))
    step = max(1, len(cal) // n)
    return cal[::step][:n]


def fit_maps(pred_h5, gt_h5, cal, n_lead, cap_global, cap_per_lead, seed=0):
    """Return (global_map, [per_lead_map]*n_lead) as (src, dst) pairs.

    The global map is fitted from `take_g`, the per-lead maps from `take_l`, each
    a stride-spread subsample of the calibration half sized to its own pixel
    budget. Holding `take_g` fixed across budgets is what makes the two
    comparable: only the per-lead sample changes, so a move in the per-lead
    column cannot be blamed on the global one having moved underneath it.
    """
    rng = np.random.default_rng(seed)
    need_ev_g = max(1, cap_global // (PIX_PER_LEAD * n_lead))
    need_ev_l = max(1, cap_per_lead // PIX_PER_LEAD)
    take_g, take_l = _take(cal, need_ev_g), _take(cal, need_ev_l)
    union = sorted(set(take_g.tolist()) | set(take_l.tolist()))

    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    Pv, Gv = fp["pred_vil"], fg["OUT_vil"]
    store = {i: (Pv[i].astype(np.float32), Gv[i].astype(np.float32))
             for i in union}
    fp.close(); fg.close()

    pg = np.concatenate([store[i][0][..., t].reshape(-1)
                         for i in take_g for t in range(n_lead)])
    gg = np.concatenate([store[i][1][..., t].reshape(-1)
                         for i in take_g for t in range(n_lead)])
    if pg.size > cap_global:
        sel = rng.choice(pg.size, size=cap_global, replace=False)
        pg, gg = pg[sel], gg[sel]
    gmap = _percentile_map(pg, gg)
    del pg, gg

    lmaps = []
    for t in range(n_lead):
        p = np.concatenate([store[i][0][..., t].reshape(-1) for i in take_l])
        g = np.concatenate([store[i][1][..., t].reshape(-1) for i in take_l])
        if p.size > cap_per_lead:
            sel = rng.choice(p.size, size=cap_per_lead, replace=False)
            p, g = p[sel], g[sel]
        lmaps.append(_percentile_map(p, g))
    return gmap, lmaps, len(take_l), len(take_g)


def _interp(x, s, t):
    flat = x.reshape(-1)
    i = torch.searchsorted(s, flat).clamp(1, len(s) - 1)
    x0, x1, y0, y1 = s[i - 1], s[i], t[i - 1], t[i]
    w = torch.where(x1 > x0, (flat - x0) / (x1 - x0 + 1e-12),
                    torch.zeros_like(flat))
    return (y0 + w * (y1 - y0)).reshape(x.shape)


def make_knobs(gmap, lmaps, device):
    gs = torch.from_numpy(gmap[0]).to(device)
    gt_ = torch.from_numpy(gmap[1]).to(device)
    ls = [torch.from_numpy(a).to(device) for a, _ in lmaps]
    lt = [torch.from_numpy(b).to(device) for _, b in lmaps]

    def ident(x):
        return x

    def qglobal(x):
        return _interp(x, gs, gt_)

    def qlead(x):
        out = torch.empty_like(x)
        for t in range(x.shape[1]):
            out[:, t] = _interp(x[:, t], ls[t], lt[t])
        return out
    return {"identity": ident, "qmap_global": qglobal, "qmap_lead": qlead}


def score(pred_h5, gt_h5, idx, knobs, device, n_lead, batch=8):
    """acc[pooling][knob][thr][lead] = [h, f, m], lead-resolved."""
    names = list(knobs)
    acc = np.zeros((len(POOLINGS), len(names), len(THRESHOLDS), n_lead, 3),
                   dtype=np.int64)
    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    Pv, Gv = fp["pred_vil"], fg["OUT_vil"]
    for s in range(0, len(idx), batch):
        sel = np.sort(idx[s:s + batch])
        p = torch.from_numpy(Pv[sel].astype(np.float32)).to(device)
        g = torch.from_numpy(Gv[sel].astype(np.float32)).to(device)
        p, g = p.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)   # (B,T,H,W)
        for ki, kn in enumerate(names):
            pk = knobs[kn](p)
            for pi, plname in enumerate(POOLINGS):
                pp, gp = pool(pk, plname), pool(g, plname)
                for ti, thr in enumerate(THRESHOLDS):
                    ph, gh = pp >= thr, gp >= thr
                    h = torch.count_nonzero(ph & gh, dim=(0, 2, 3))
                    f = torch.count_nonzero(ph & ~gh, dim=(0, 2, 3))
                    m = torch.count_nonzero(~ph & gh, dim=(0, 2, 3))
                    trip = torch.stack([h, f, m], -1).cpu().numpy()
                    acc[pi, ki, ti] += trip
        if (s // batch) % 20 == 0:
            print(f"  {s}/{len(idx)}", flush=True)
    fp.close(); fg.close()
    return acc, names


def csi_bias(h, f, m):
    return (h / max(h + f + m, 1), (h + f) / max(h + m, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", choices=["matched", "equal"],
                    default="matched")
    ap.add_argument("--cap-global", type=int, default=60_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0,
                    help="smoke test: score only this many eval events")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(a.pred, "r") as f:
        N, n_lead = f["pred_vil"].shape[0], f["pred_vil"].shape[3]
    cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)

    cap_l = (a.cap_global // n_lead if a.budget == "matched"
             else a.cap_global)
    gmap, lmaps, n_take, n_g = fit_maps(a.pred, a.gt, cal, n_lead,
                                        a.cap_global, cap_l, a.seed)
    print(f"{a.name}: N={N} leads={n_lead} budget={a.budget} "
          f"events_read={n_take} events_for_global={n_g} "
          f"px_per_lead_map<={cap_l}", flush=True)

    if a.max_eval:
        ev = ev[:a.max_eval]
    knobs = make_knobs(gmap, lmaps, dev)
    acc, names = score(a.pred, a.gt, ev, knobs, dev, n_lead)

    out = {"name": a.name, "budget": a.budget, "seed": a.seed,
           "n_lead": n_lead, "n_eval": len(ev), "knobs": names,
           "thresholds": THRESHOLDS, "poolings": POOLINGS,
           "cap_global": a.cap_global, "cap_per_lead": cap_l,
           "table": [], "per_lead": []}
    for pi, plname in enumerate(POOLINGS):
        for ti, thr in enumerate(THRESHOLDS):
            row = {"pooling": plname, "threshold": thr}
            for ki, kn in enumerate(names):
                h, f, m = acc[pi, ki, ti].sum(0)
                c, b = csi_bias(int(h), int(f), int(m))
                row[f"csi_{kn}"] = c
                row[f"bias_{kn}"] = b
            out["table"].append(row)
    for ti, thr in enumerate(THRESHOLDS):
        pi = POOLINGS.index("max16")
        for L in range(n_lead):
            r = {"threshold": thr, "lead": L}
            for ki, kn in enumerate(names):
                h, f, m = acc[pi, ki, ti, L]
                c, b = csi_bias(int(h), int(f), int(m))
                r[f"csi_{kn}"] = c
                r[f"bias_{kn}"] = b
            out["per_lead"].append(r)
    a.out.write_text(json.dumps(out, indent=2))

    print(f"\n{'pool':>6} {'thr':>5} {'ident':>8} {'bias':>7} "
          f"{'global':>8} {'bias':>7} {'perlead':>8} {'bias':>7}")
    for r in out["table"]:
        print(f"{r['pooling']:>6} {r['threshold']:>5} "
              f"{r['csi_identity']:8.4f} {r['bias_identity']:7.3f} "
              f"{r['csi_qmap_global']:8.4f} {r['bias_qmap_global']:7.3f} "
              f"{r['csi_qmap_lead']:8.4f} {r['bias_qmap_lead']:7.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
