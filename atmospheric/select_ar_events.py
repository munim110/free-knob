"""Select the atmospheric-river time steps the downscaling dataset is built on.

What counts as an event is not our choice. It comes from a published global
atmospheric-river catalogue computed on ERA5, so the selection rule is a
community algorithm with its own citation rather than a threshold we picked to
suit the experiment. A 6-hourly step enters the record when the catalogue marks
any grid cell inside the study box as belonging to an AR object, summed over the
catalogue's ensemble and level axes.

Why the query span and the record length differ
-----------------------------------------------
The catalogue itself covers 1940 to 2024 and the query below runs from 2010, but
the paired record is shorter because the target channel constrains it. Himawari-8
B08 became operational in July 2015, and Himawari-9 took over primary operations
in December 2022, so usable pairs exist only from 2015 onward. The intersection
of this event list with target availability is what produces the nine-year,
1,500-scene record the paper reports; this query alone does not.

The study box below is the single block to edit if the domain is restated.
"""
import argparse
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402


# Study box for the downscaling domain: 7 degrees of latitude by 5 of longitude,
# about 780 km by 510 km, over a tropical monsoon-influenced coastal and marine
# sector. Stated here once so the domain can be restated in one edit.
LAT_MIN, LAT_MAX = 20.0, 27.0
LON_MIN, LON_MAX = 88.0, 93.0

# The catalogue query span. The usable record is the intersection of this with
# target availability, which starts in July 2015.
START_DATE, END_DATE = "2010-01-01", "2023-12-31"


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, required=True,
                    help="the global AR catalogue NetCDF computed on ERA5")
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    ap.add_argument("--lat-min", type=float, default=LAT_MIN)
    ap.add_argument("--lat-max", type=float, default=LAT_MAX)
    ap.add_argument("--lon-min", type=float, default=LON_MIN)
    ap.add_argument("--lon-max", type=float, default=LON_MAX)
    ap.add_argument("--var", default="shapemap",
                    help="catalogue variable carrying the AR object mask")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.ATMOS_DATA) + "/ar_event_times.txt"))
    return ap.parse_args()


def main():
    args = parse_args()
    import xarray as xr

    P.require(args.catalog, "AR catalogue NetCDF")
    ds = xr.open_dataset(args.catalog)

    # Latitude is stored in descending order in the catalogue, so the slice runs
    # from the northern edge to the southern one.
    regional = ds.sel(
        time=slice(args.start, args.end),
        lat=slice(args.lat_max, args.lat_min),
        lon=slice(args.lon_min, args.lon_max),
    )

    mask = regional[args.var]
    reduce_over = [d for d in ("lat", "lon", "ens", "lev") if d in mask.dims]
    present = mask.sum(dim=reduce_over) > 0
    times = regional["time"].where(present, drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamps = [str(t)[:19].replace(" ", "T") for t in times.values]
    args.out.write_text("\n".join(stamps) + "\n", encoding="utf-8")

    print(f"box: lat [{args.lat_min}, {args.lat_max}], "
          f"lon [{args.lon_min}, {args.lon_max}]")
    print(f"span: {args.start} to {args.end}")
    print(f"{len(stamps)} six-hourly steps carry an AR object in the box")
    print(f"wrote {args.out}")
    print("Pair these against target availability before building the dataset; "
          "steps before July 2015 have no B08 target.")


if __name__ == "__main__":
    main()
