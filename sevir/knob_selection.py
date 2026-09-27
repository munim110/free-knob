"""Four selection rules for the knob, at the SEVIR cells the headlines are quoted at.

Appendix L asks this on crowd density and never carried it to SEVIR. Appendix O
measured a per-horizon transform under one rule -- carry each lead's forecast
marginal onto that lead's observed marginal -- and found it erases less than the
global map. Both gaps are the same missing experiment: the knob's payoff is a
function of how the knob is *chosen*, and the paper reports one choice.

This crosses the two axes in a single accumulation.

  granularity   global    one member of G for the whole field
                per-lead  one member per forecast horizon

  rule          bias      minimise |B - 1| on the calibration half. The paper's
                          default. Conservative: it cannot tune for the score we
                          report, and it penalises an arm whose bias is long.
                    csi   maximise CSI on the calibration half. Gives each arm
                          its best shot within G, but consumes paired labels
                          rather than the observed marginal alone, so it sits
                          outside the "unpaired statistics only" description the
                          rest of the paper claims and is a robustness check
                          rather than the control.

Every rule sees the calibration half only. `oracle` is the best evaluation-half
CSI over the family; it is not a rule, since nothing could select it without
seeing test, and bounds what the four leave behind.

Counts are accumulated once per (pooling, knob, threshold, lead) on each half, so
all four rules and the oracle come from one pass over the prediction fields.
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
import paths as P                                              # noqa: E402
from pooling_control import (THRESHOLDS, SHIFTS, MULS, pool,    # noqa: E402
                             fit_global_qmap)


def build_knobs(pred, gt, cal, device):
    knobs = {"identity": lambda x: x}
    for d in SHIFTS:
        knobs[f"shift={d}"] = (lambda d: lambda x: x + d)(d)
    for g in MULS:
        knobs[f"mul={g}"] = (lambda g: lambda x: x * g)(g)
    src, dst = fit_global_qmap(pred, gt, cal)
    s = torch.from_numpy(src).to(device)
    t = torch.from_numpy(dst).to(device)

    def _q(x):
        flat = x.reshape(-1)
        i = torch.searchsorted(s, flat).clamp(1, len(s) - 1)
        x0, x1, y0, y1 = s[i - 1], s[i], t[i - 1], t[i]
        w = torch.where(x1 > x0, (flat - x0) / (x1 - x0 + 1e-12),
                        torch.zeros_like(flat))
        return (y0 + w * (y1 - y0)).reshape(x.shape)
    knobs["qmap_global"] = _q
    return knobs


def accumulate(pred_h5, gt_h5, idx, knobs, poolings, device, n_lead, batch=8):
    """counts[pooling, knob, threshold, lead] = (h, f, m), summed over events."""
    names = list(knobs)
    acc = np.zeros((len(poolings), len(names), len(THRESHOLDS), n_lead, 3),
                   dtype=np.int64)
    fp, fg = h5py.File(pred_h5, "r"), h5py.File(gt_h5, "r")
    Pv, Gv = fp["pred_vil"], fg["OUT_vil"]
    for s0 in range(0, len(idx), batch):
        sel = np.sort(idx[s0:s0 + batch])
        p = torch.from_numpy(Pv[sel].astype(np.float32)).to(device)
        g = torch.from_numpy(Gv[sel].astype(np.float32)).to(device)
        p, g = p.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        for ki, kn in enumerate(names):
            pk = knobs[kn](p)
            for pi, plname in enumerate(poolings):
                pp, gp = pool(pk, plname), pool(g, plname)
                for ti, thr in enumerate(THRESHOLDS):
                    ph, gh = pp >= thr, gp >= thr
                    h = torch.count_nonzero(ph & gh, dim=(0, 2, 3))
                    f = torch.count_nonzero(ph & ~gh, dim=(0, 2, 3))
                    m = torch.count_nonzero(~ph & gh, dim=(0, 2, 3))
                    acc[pi, ki, ti] += torch.stack([h, f, m], -1).cpu().numpy()
        if (s0 // batch) % 25 == 0:
            print(f"  {s0}/{len(idx)}", flush=True)
    fp.close(); fg.close()
    return acc, names


def _cb(trip):
    h, f, m = (int(trip[..., 0]), int(trip[..., 1]), int(trip[..., 2]))
    return (h / max(h + f + m, 1), (h + f) / max(h + m, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poolings", type=str, default="none,max16")
    a = ap.parse_args()

    poolings = [p for p in a.poolings.split(",") if p]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(a.pred, "r") as f:
        N, n_lead = f["pred_vil"].shape[0], f["pred_vil"].shape[3]
    cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)

    knobs = build_knobs(a.pred, a.gt, cal, dev)
    print(f"{a.name}: N={N} leads={n_lead} knobs={len(knobs)} "
          f"poolings={poolings}", flush=True)
    C, names = accumulate(a.pred, a.gt, cal, knobs, poolings, dev, n_lead)
    print("  calibration half accumulated", flush=True)
    E, _ = accumulate(a.pred, a.gt, ev, knobs, poolings, dev, n_lead)
    print("  evaluation half accumulated", flush=True)

    out = {"name": a.name, "n_calib": len(cal), "n_eval": len(ev),
           "n_lead": n_lead, "poolings": poolings, "thresholds": THRESHOLDS,
           "knobs": names, "table": []}
    for pi, plname in enumerate(poolings):
        for ti, thr in enumerate(THRESHOLDS):
            c_tot = C[pi, :, ti].sum(1)            # (knob, 3) over leads
            e_tot = E[pi, :, ti].sum(1)
            row = {"pooling": plname, "threshold": thr}
            row["csi_uncal"], row["bias_uncal"] = _cb(e_tot[names.index("identity")])

            kb = min(range(len(names)), key=lambda k: abs(_cb(c_tot[k])[1] - 1.0))
            kc = max(range(len(names)), key=lambda k: _cb(c_tot[k])[0])
            row["knob_global_bias"], row["knob_global_csi"] = names[kb], names[kc]
            row["csi_global_bias"], row["bias_global_bias"] = _cb(e_tot[kb])
            row["csi_global_csi"], row["bias_global_csi"] = _cb(e_tot[kc])

            for tag, key in (("bias", None), ("csi", None)):
                picks, agg = [], np.zeros(3, dtype=np.int64)
                for L in range(n_lead):
                    cl = C[pi, :, ti, L]
                    if tag == "bias":
                        k = min(range(len(names)),
                                key=lambda k: abs(_cb(cl[k])[1] - 1.0))
                    else:
                        k = max(range(len(names)), key=lambda k: _cb(cl[k])[0])
                    picks.append(names[k])
                    agg += E[pi, k, ti, L]
                row[f"csi_perlead_{tag}"], row[f"bias_perlead_{tag}"] = _cb(agg)
                row[f"knobs_perlead_{tag}"] = picks

            row["csi_oracle"] = max(_cb(e_tot[k])[0] for k in range(len(names)))
            out["table"].append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))

    print(f"\n{'pool':>6}{'thr':>5}{'uncal':>9}{'gBias':>9}{'gCSI':>9}"
          f"{'lBias':>9}{'lCSI':>9}{'oracle':>9}")
    for r in out["table"]:
        print(f"{r['pooling']:>6}{r['threshold']:>5}{r['csi_uncal']:9.4f}"
              f"{r['csi_global_bias']:9.4f}{r['csi_global_csi']:9.4f}"
              f"{r['csi_perlead_bias']:9.4f}{r['csi_perlead_csi']:9.4f}"
              f"{r['csi_oracle']:9.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
