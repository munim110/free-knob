"""Free-knob audit of global 24-h precipitation forecasts from WeatherBench 2.

Public forecasts (no inference): GraphCast (ERA5- and HRES-initialised), FuXi,
IFS HRES and the IFS ensemble mean, all at 0.25 deg, scored against ERA5
24-h accumulated precipitation, 2020, 00 UTC initialisations.

Event: 24-h precipitation >= tau mm in a block (pixel, or 1 deg = 4x4 block
under max or mean pooling). Counts are area-weighted by cos(latitude).

Control: per (model, lead, pooling, tau), the forecast threshold whose
area-weighted exceedance on the calibration days equals the observed one.
It is fitted on pooled block values, so it is exactly a per-model threshold
under every pooling (no block can be reordered). Calibration days are the
even-indexed initialisation dates, evaluation days the odd-indexed ones.

Writes <out>/wb2_<model>.json (summary) and wb2_<model>.npz (per-day
evaluation counts, for the day bootstrap in wb2_precip_report.py).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GS = "gs://weatherbench2/datasets/"
MODELS = {
    "graphcast": (GS + "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr",
                  "total_precipitation_24hr"),
    "graphcast_oper": (GS + "graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr",
                       "total_precipitation_24hr"),
    "hres": (GS + "hres/2016-2022-0012-1440x721.zarr", "total_precipitation_24hr"),
    "ens_mean": (GS + "ifs_ens/2018-2022-1440x721_mean.zarr", "total_precipitation_24hr"),
    "fuxi": (GS + "fuxi/2020-1440x721.zarr", "total_precipitation_24hr_from_6hr"),
}
ERA5 = (GS + "era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        "total_precipitation_24hr")
POOLINGS = ("none", "max4", "avg4")
FLOOR = 0.2  # mm; calibration values below this are not stored (flagged if needed)
SO = {"token": "anon"}


def std(da):
    """Rename to (time, [prediction_timedelta,] lat, lon), lat ascending."""
    da = da.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.dims})
    return da.sortby("lat")


def pool(x):
    """x: (720, 1440) in mm -> dict pooling -> 2-D array."""
    b = x.reshape(180, 4, 360, 4)
    return {"none": x, "max4": b.max(axis=(1, 3)), "avg4": b.mean(axis=(1, 3))}


def lat_weights(lat):
    # float64: a float32 running sum over ~1e8 pixels stalls and corrupts the fit
    w = np.cos(np.deg2rad(lat[1:].astype(np.float64)))  # drop the -90 row (weight 0) -> 720 rows
    return {"none": w, "max4": w.reshape(180, 4).mean(1), "avg4": w.reshape(180, 4).mean(1)}


# metres -> mm; FuXi's WB2 field is already in mm (no units attribute; checked
# against ERA5's global mean of about 2.4 mm/day)
SCALE = {"fuxi": 1.0}


def to_mm(a, scale=1000.0):
    return np.clip(np.asarray(a, dtype=np.float32)[1:, :] * scale, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--leads", nargs="+", type=int, default=[24, 72, 120])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2020-12-15")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[1, 10, 25, 50])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/revision/wb2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(args.start, args.end, freq="D")
    leads = [np.timedelta64(h, "h") for h in args.leads]
    thr = np.array(args.thresholds, dtype=np.float32)
    calib_idx = [i for i in range(len(dates)) if i % 2 == 0]
    eval_idx = [i for i in range(len(dates)) if i % 2 == 1]

    # ---- ERA5 truth, pooled, cached once for every valid time ----
    era = std(xr.open_zarr(ERA5[0], storage_options=SO)[ERA5[1]])
    lat = era.lat.values
    W = lat_weights(lat)
    valid = sorted({d + np.timedelta64(h, "h") for d in dates for h in args.leads})
    t0 = time.time()

    def get_truth(v):
        return v, pool(to_mm(era.sel(time=v).values))

    with ThreadPoolExecutor(args.workers) as ex:
        truth = dict(ex.map(get_truth, valid))
    print(f"ERA5: {len(truth)} valid times in {time.time() - t0:.0f}s", flush=True)

    for name in args.models:
        path, var = MODELS[name]
        fc = std(xr.open_zarr(path, storage_options=SO)[var])
        assert np.allclose(fc.lat.values, lat), f"{name}: latitude grid differs from ERA5"

        def get_fc(i):
            x = fc.sel(time=dates[i], prediction_timedelta=leads).values  # (L, 721, 1440)
            return i, [pool(to_mm(x[j], SCALE.get(name, 1000.0))) for j in range(len(leads))]

        L, P, T = len(leads), len(POOLINGS), len(thr)
        # calibration pass: store pooled forecast values >= FLOOR with row indices
        cal_vals = {(j, p): [] for j in range(L) for p in POOLINGS}
        cal_rows = {(j, p): [] for j in range(L) for p in POOLINGS}
        obs_w = np.zeros((L, P, T))
        fc_w_raw = np.zeros((L, P, T))
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            for n, (i, F) in enumerate(ex.map(get_fc, calib_idx)):
                for j, lead in enumerate(leads):
                    O = truth[dates[i] + lead]
                    for pi, p in enumerate(POOLINGS):
                        f, o, w = F[j][p], O[p], W[p][:, None]
                        m = f >= FLOOR
                        cal_vals[(j, p)].append(f[m])
                        cal_rows[(j, p)].append(np.nonzero(m)[0].astype(np.int16))
                        for ti, tau in enumerate(thr):
                            obs_w[j, pi, ti] += (w * (o >= tau)).sum()
                            fc_w_raw[j, pi, ti] += (w * (f >= tau)).sum()
                if n % 25 == 0:
                    print(f"{name} calib {n}/{len(calib_idx)} {time.time() - t0:.0f}s", flush=True)
        thr_cal = np.zeros((L, P, T), dtype=np.float32)
        flag = np.zeros((L, P, T), dtype=bool)
        for j in range(L):
            for pi, p in enumerate(POOLINGS):
                v = np.concatenate(cal_vals.pop((j, p)))
                r = np.concatenate(cal_rows.pop((j, p)))
                order = np.argsort(-v, kind="stable")
                v, cw = v[order], np.cumsum(W[p][r[order]])
                for ti in range(T):
                    k = int(np.searchsorted(cw, obs_w[j, pi, ti]))
                    if k >= len(v):
                        flag[j, pi, ti], thr_cal[j, pi, ti] = True, FLOOR
                    else:
                        thr_cal[j, pi, ti] = v[k]
                del v, cw, r, order

        # evaluation pass: per-day weighted (hits, false alarms, misses), raw and calibrated
        counts = np.zeros((len(eval_idx), L, P, T, 2, 3))
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            for n, (i, F) in enumerate(ex.map(get_fc, eval_idx)):
                for j, lead in enumerate(leads):
                    O = truth[dates[i] + lead]
                    for pi, p in enumerate(POOLINGS):
                        f, o, w = F[j][p], O[p], W[p][:, None]
                        for ti, tau in enumerate(thr):
                            ob = o >= tau
                            for k, t in enumerate((tau, thr_cal[j, pi, ti])):
                                fb = f >= t
                                counts[n, j, pi, ti, k] = ((w * (fb & ob)).sum(),
                                                           (w * (fb & ~ob)).sum(),
                                                           (w * (~fb & ob)).sum())
                if n % 25 == 0:
                    print(f"{name} eval {n}/{len(eval_idx)} {time.time() - t0:.0f}s", flush=True)

        tot = counts.sum(0)
        h, fa, mi = tot[..., 0], tot[..., 1], tot[..., 2]
        csi = h / np.maximum(h + fa + mi, 1e-12)
        bias = (h + fa) / np.maximum(h + mi, 1e-12)
        table = []
        for j, lh in enumerate(args.leads):
            for pi, p in enumerate(POOLINGS):
                for ti, tau in enumerate(thr):
                    table.append({"lead_h": lh, "pooling": p, "threshold_mm": float(tau),
                                  "csi_raw": float(csi[j, pi, ti, 0]), "bias_raw": float(bias[j, pi, ti, 0]),
                                  "csi_cal": float(csi[j, pi, ti, 1]), "bias_cal": float(bias[j, pi, ti, 1]),
                                  "fc_threshold_cal_mm": float(thr_cal[j, pi, ti]),
                                  "calib_bias_raw": float(fc_w_raw[j, pi, ti] / max(obs_w[j, pi, ti], 1e-12)),
                                  "obs_rate_eval": float((h + mi)[j, pi, ti, 0] /
                                                         (W[p].sum() * (360 * 4 if p == "none" else 360) * len(eval_idx))),
                                  "floor_flag": bool(flag[j, pi, ti])})
        meta = {"model": name, "path": path, "var": var, "truth": ERA5[0], "leads_h": args.leads,
                "poolings": POOLINGS, "thresholds_mm": args.thresholds,
                "dates": [str(d.date()) for d in dates], "n_calib": len(calib_idx), "n_eval": len(eval_idx)}
        (out / f"wb2_{name}.json").write_text(json.dumps({**meta, "table": table}, indent=1))
        np.savez_compressed(out / f"wb2_{name}.npz", counts=counts,
                            eval_dates=np.array([str(dates[i].date()) for i in eval_idx]))
        print(f"{name}: done", flush=True)
        for r in table:
            if r["pooling"] == "none":
                print(f"  {r['lead_h']:4d}h {r['threshold_mm']:5.0f}mm  CSI {r['csi_raw']:.4f} B {r['bias_raw']:.3f}"
                      f"  -> CSI {r['csi_cal']:.4f} B {r['bias_cal']:.3f}  (t={r['fc_threshold_cal_mm']:.2f}mm)",
                      flush=True)


if __name__ == "__main__":
    main()
