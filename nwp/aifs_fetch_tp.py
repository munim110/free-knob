"""Fetch 24-h precipitation from ECMWF open data (AIFS and IFS) and ERA5 truth, no inference.

Window: 2025-08-28 .. 2026-05-11, 00 UTC initialisations. Every system is version-constant
over it: AIFS Single 1.1.0 (operational from 27 Aug 2025), AIFS ENS v1 (from 1 Jul 2025),
IFS cycle 49r1; AIFS v2 and IFS 50r1 replaced them on 12 May 2026.

Arms (ECMWF open data, 0.25 deg, Google Cloud mirror of the official archive):
  aifs_single   aifs-single/oper fc
  ifs_hres      ifs/oper fc
  aifs_ens_m1   aifs-ens/enfo pf, member 1
  ifs_ens_m1    ifs/enfo pf, member 1
Only the total-precipitation GRIB message is downloaded (byte range from the .index file).
tp is accumulated from the start of the forecast (IFS in m, AIFS in kg m**-2, read from
each message's units); the 24-h total ending at lead L is
tp(L) - tp(L - 24).

Truth: ERA5 hourly total precipitation (ARCO-ERA5 zarr, final ERA5 up to 2026-06-30); the
24-h total ending at V is the sum of the hourly accumulations valid V-23h .. V.

Output (in --out): <arm>.npy float32 (n_dates, n_leads, 721, 1440) in mm, latitude 90 .. -90,
longitude 0 .. 359.75 (the ERA5 grid); NaN where a file is missing. era5.npy is laid out the
same way. meta.json records dates, leads, sources and missing files.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

G = "https://storage.googleapis.com/ecmwf-open-data"
ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARMS = {
    "aifs_single": ("aifs-single", "oper", "fc", None),
    "ifs_hres": ("ifs", "oper", "fc", None),
    "aifs_ens_m1": ("aifs-ens", "enfo", "pf", 1),
    "ifs_ens_m1": ("ifs", "enfo", "ef", 1),
}
NY, NX = 721, 1440


def get(url, rng=None, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(60, 2 ** k))
        except Exception:
            time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"failed: {url}")


def decode_tp(msg):
    """One GRIB2 message -> (721, 1440) mm on the ERA5 grid orientation."""
    import eccodes
    h = eccodes.codes_new_from_message(msg)
    try:
        assert eccodes.codes_get(h, "shortName") == "tp"
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        jpos = eccodes.codes_get(h, "jScansPositively")
        units = eccodes.codes_get(h, "units")
        v = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
    finally:
        eccodes.codes_release(h)
    # IFS encodes tp in m, AIFS in kg m**-2 (= mm); return mm
    scale = {"m": 1000.0, "kg m**-2": 1.0}[units]
    v = v * np.float32(scale)
    assert (nj, ni) == (NY, NX), (nj, ni)
    if jpos:
        v, lat0 = v[::-1], -lat0
    assert abs(lat0 - 90) < 1e-6, lat0
    lon0 = lon0 % 360
    if abs(lon0 - 180) < 1e-6:          # -180 .. 179.75 -> 0 .. 359.75
        v = np.roll(v, NX // 2, axis=1)
    else:
        assert abs(lon0) < 1e-6, lon0
    return v


def fetch_field(arm, date, step):
    model, stream, ftype, number = ARMS[arm]
    d = date.strftime("%Y%m%d")
    base = f"{G}/{d}/00z/{model}/0p25/{stream}/{d}000000-{step}h-{stream}-{ftype}"
    ix = get(base + ".index")
    if ix is None:
        return None
    for line in ix.decode().splitlines():
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        if e.get("param") != "tp":
            continue
        if number is not None and int(e.get("number", -1)) != number:
            continue
        msg = get(base + ".grib2", (e["_offset"], e["_offset"] + e["_length"] - 1))
        return None if msg is None else decode_tp(msg)
    return None


def era5_index():
    import numcodecs
    raw = get(ARCO + "/time/0")
    t = np.frombuffer(numcodecs.Blosc().decode(raw), dtype="<i8")
    t0 = pd.Timestamp("1900-01-01") + pd.Timedelta(hours=int(t[0]))
    assert np.all(np.diff(t) == 1)
    return t0


def era5_hour(i):
    import numcodecs
    raw = get(f"{ARCO}/total_precipitation/{i}.0.0")
    return np.frombuffer(numcodecs.Blosc().decode(raw), dtype="<f4").reshape(NY, NX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-28")
    ap.add_argument("--end", default="2026-05-11")
    ap.add_argument("--leads", nargs="+", type=int, default=[24, 72, 144])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="aifs_tp")
    ap.add_argument("--skip-era5", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(args.start, args.end, freq="D")
    leads = args.leads
    steps = sorted({s for L in leads for s in (L, L - 24) if s > 0})
    meta = {"dates": [str(d.date()) for d in dates], "leads_h": leads, "units": "mm",
            "grid": "lat 90..-90, lon 0..359.75, 0.25 deg", "arms": ARMS, "era5": ARCO,
            "missing": {}}

    with ThreadPoolExecutor(args.workers) as ex:
        for arm in args.arms:
            path = out / f"{arm}.npy"
            A = np.lib.format.open_memmap(path, "w+", np.float32, (len(dates), len(leads), NY, NX))
            missing = []
            t0 = time.time()

            def one(i):
                f = {s: fetch_field(arm, dates[i], s) for s in steps}
                return i, f

            for n, (i, f) in enumerate(ex.map(one, range(len(dates)))):
                for j, L in enumerate(leads):
                    a, b = f.get(L), (np.zeros((NY, NX), np.float32) if L == 24 else f.get(L - 24))
                    if a is None or b is None:
                        A[i, j] = np.nan
                        missing.append([str(dates[i].date()), L])
                    else:
                        A[i, j] = np.clip(a - b, 0, None)
                if n % 20 == 0:
                    print(f"{arm} {n}/{len(dates)} {time.time() - t0:.0f}s missing={len(missing)}", flush=True)
            A.flush()
            del A
            meta["missing"][arm] = missing
            print(f"{arm}: done, {len(missing)} missing", flush=True)
            (out / "meta.json").write_text(json.dumps(meta, indent=1))

        if not args.skip_era5:
            t_first = era5_index()
            valid = sorted({d + pd.Timedelta(hours=L) for d in dates for L in leads})
            day = {}
            t0 = time.time()

            def one_day(v):
                idx = [int((v - pd.Timedelta(hours=k) - t_first) / pd.Timedelta(hours=1)) for k in range(24)]
                s = np.zeros((NY, NX), np.float64)
                for i in idx:
                    s += era5_hour(i)
                return v, (s * 1000.0).astype(np.float32)

            for n, (v, s) in enumerate(ex.map(one_day, valid)):
                day[v] = s
                if n % 20 == 0:
                    print(f"era5 {n}/{len(valid)} {time.time() - t0:.0f}s", flush=True)
            E = np.lib.format.open_memmap(out / "era5.npy", "w+", np.float32, (len(dates), len(leads), NY, NX))
            for i, d in enumerate(dates):
                for j, L in enumerate(leads):
                    E[i, j] = day[d + pd.Timedelta(hours=L)]
            E.flush()
            meta["era5_first_time"] = str(t_first)
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print("era5: done", flush=True)


if __name__ == "__main__":
    main()
