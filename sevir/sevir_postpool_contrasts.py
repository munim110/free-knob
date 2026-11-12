"""The 450 SEVIR contrasts under the paper's map and under the map fitted after pooling.

Reads results/revision/postpool/sevir_postpool_<arm>.json for the six arms (written by
sevir_pool_then_calibrate.py on the rerun predictions) and rebuilds every
pairwise contrast with the definitions of sevir/published_contrasts.py:
raw = CSI_b - CSI_a, cal = the same after each arm's own map, distortion =
raw - cal, calibration gap = |log bias_a| - |log bias_b|, flip = sign change.

Three controls are compared on the same predictions:
  pre         the paper's map (fitted on pixels, applied before pooling);
  post_exact  the exact empirical quantile map fitted on pooled values;
  post_grid   the same with the paper's 1001-knot convention.
"""
import json
import math
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402
RES = P.RESULTS / "revision" / "postpool"
ARMS = ["persistence", "pysteps", "earthformer", "cascast_det",
        "cascast_cascade", "cascast_cascade_cfg1"]
PUBLISHED = {("cascast_det", "cascast_cascade"), ("earthformer", "cascast_cascade"),
             ("persistence", "earthformer")}
KINDS = ("pre", "post_exact", "post_grid")


def dev(b):
    return abs(math.log(b)) if b and b > 0 else None


def main():
    A = {}
    for a in ARMS:
        d = json.loads((RES / f"sevir_postpool_{a}.json").read_text())
        A[a] = {(r["pooling"], r["threshold"]): r for r in d["table"]}
    rows = []
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            for key in A[a]:
                ra, rb = A[a][key], A[b][key]
                da, db = dev(ra["bias_raw"]), dev(rb["bias_raw"])
                if da is None or db is None:
                    continue
                raw = rb["csi_raw"] - ra["csi_raw"]
                row = {"a": a, "b": b, "pooling": key[0], "threshold": key[1],
                       "devgap": da - db, "raw": raw,
                       "published": (a, b) in PUBLISHED}
                for k in KINDS:
                    cal = rb[f"csi_{k}"] - ra[f"csi_{k}"]
                    row[f"cal_{k}"] = cal
                    row[f"distort_{k}"] = raw - cal
                    row[f"flip_{k}"] = (raw > 0) != (cal > 0)
                rows.append(row)

    subsets = [("all", rows),
               ("max or none", [r for r in rows if not r["pooling"].startswith("avg")]),
               ("avg only", [r for r in rows if r["pooling"].startswith("avg")]),
               ("max16", [r for r in rows if r["pooling"] == "max16"]),
               ("published", [r for r in rows if r["published"]])]
    out = {"n": len(rows), "stats": {}}
    lines = ["# SEVIR contrasts: map before pooling vs map after pooling", "",
             f"{len(rows)} contrasts over {len(ARMS)} rerun arms. r and ρ are between the "
             "calibration gap and the distortion; flips are sign reversals.", "",
             "| subset | n | " + " | ".join(f"{k}: r / ρ / flips" for k in KINDS) + " |",
             "|---|---|" + "---|" * len(KINDS)]
    for tag, sub in subsets:
        x = [r["devgap"] for r in sub]
        cells, st = [], {"n": len(sub)}
        for k in KINDS:
            y = [r[f"distort_{k}"] for r in sub]
            pr, sr = pearsonr(x, y).statistic, spearmanr(x, y).statistic
            fl = sum(r[f"flip_{k}"] for r in sub)
            st[k] = {"pearson": pr, "spearman": sr, "flips": fl}
            cells.append(f"{pr:+.3f} / {sr:+.3f} / {fl}")
        out["stats"][tag] = st
        lines.append(f"| {tag} | {len(sub)} | " + " | ".join(cells) + " |")
    (RES / "sevir_postpool_contrasts.json").write_text(json.dumps({**out, "rows": rows}, indent=1))
    (RES / "SEVIR_POSTPOOL_CONTRASTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
