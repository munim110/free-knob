"""Pairwise contrasts between SEVIR arms, before and after both sides are calibrated.

decomposition_report.py measures how much of CasCast's reported cascade gain
survives a free monotone transform. That is a single contrast from a single
paper. This script runs the same comparison over every ordered pair of the arms
we hold, which gives a distribution rather than one number and allows the
direction of the effect to be measured.

All arms are evaluated by one pipeline on the official test split:

    persistence            last observed frame, an observed radar field
    pysteps                Lucas-Kanade advection, the operational reference
    earthformer            authors' released `earthformer_sevir.pt`
    cascast_det            CasCast's released deterministic backbone
    cascast_cascade        CasCast's released cascade at cfg=2, as their eval script runs it
    cascast_cascade_cfg1   the same cascade sampled without guidance

For a pair (A, B) at one (pooling, threshold) cell:

    raw       = CSI_B - CSI_A                     the contrast a results table prints
    cal       = CSI_B^g - CSI_A^g                 the same contrast after one global
                                                  quantile map is fitted per arm on the
                                                  calibration half and held fixed
    distort   = raw - cal

`distort` equals knobgain_A - knobgain_B by construction, so it is not independent
evidence for the mechanism. It measures the size of the consequence: how far a
published contrast moves once the confound is removed. The claim under test is
the predictive one,

    H: distort is ordered by the pair's calibration gap, dev(A) - dev(B),
       where dev = |log bias| is the distance of an arm from bias 1.

H can fail while the per-arm mechanism still holds, since two arms miscalibrated
in the same direction cancel. The correlation is reported whatever its value.

Sign flips are counted separately. A cell where raw and cal disagree in sign is a
cell where calibrating both arms reverses which model the table calls better.
"""
import argparse
import json
import math
from pathlib import Path

POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]
THRESHOLDS = [16, 74, 133, 160, 181, 219]

# Display labels for the arms.
ARMS = {
    "persistence": "persistence",
    "pysteps": "pysteps (advection)",
    "earthformer": "EarthFormer",
    "cascast_det": "CasCast backbone",
    "cascast_cascade": "CasCast cascade",
    "cascast_cascade_cfg1": "CasCast cascade (no guidance)",
}

# Pairs corresponding to a comparison made in a published paper, with the source.
# Every other pair is a contrast we construct. Both are reported and labelled.
PUBLISHED = {
    ("cascast_det", "cascast_cascade"):
        "CasCast Tab. 4/5, their own backbone-vs-cascade ablation",
    ("earthformer", "cascast_cascade"):
        "CasCast Tab. 4, EarthFormer as a baseline it reports beating",
    ("persistence", "earthformer"):
        "EarthFormer Tab. 3 reports persistence as a baseline",
}

# Both arms here are the same cuboid-transformer architecture. CasCast's
# `networks/earthformer_xy.py` is EarthFormer's attention code, and the two
# checkpoints come from two groups after two separate trainings. This pair is kept
# apart from PUBLISHED because it measures reproducibility rather than architecture.
SAME_ARCH = {
    ("earthformer", "cascast_det"):
        "one architecture, two independent trainings, two released checkpoints",
}


def dev(bias):
    """Distance from a frequency bias of 1, symmetric in over- and under-forecasting."""
    if bias is None or bias <= 0:
        return None
    return abs(math.log(bias))


def load(res, name):
    d = json.loads((res / f"sevir_poolingq_{name}.json").read_text())
    return {(r["pooling"], r["threshold"]): r for r in d["table"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("results/sevir_published_contrasts.md"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/sevir_published_contrasts.json"))
    args = ap.parse_args()

    A = {}
    for name in ARMS:
        p = args.results / f"sevir_poolingq_{name}.json"
        if p.exists():
            A[name] = load(args.results, name)
    names = [n for n in ARMS if n in A]

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for pl in POOLINGS:
                for th in THRESHOLDS:
                    ra, rb = A[a].get((pl, th)), A[b].get((pl, th))
                    if ra is None or rb is None:
                        continue
                    ua, ub = ra["csi_uncal"], rb["csi_uncal"]
                    ca, cb = ra["csi_qmap_global"], rb["csi_qmap_global"]
                    da, db = dev(ra["bias_uncal"]), dev(rb["bias_uncal"])
                    if da is None or db is None:
                        continue
                    raw, cal = ub - ua, cb - ca
                    rows.append({
                        "a": a, "b": b, "pooling": pl, "threshold": th,
                        "csi_a": ua, "csi_b": ub,
                        "csi_a_cal": ca, "csi_b_cal": cb,
                        "bias_a": ra["bias_uncal"], "bias_b": rb["bias_uncal"],
                        "dev_a": da, "dev_b": db, "devgap": da - db,
                        "raw": raw, "cal": cal, "distort": raw - cal,
                        "rel_raw": (ub - ua) / ua if ua > 0 else None,
                        "rel_cal": (cb - ca) / ca if ca > 0 else None,
                        "flip": (raw > 0) != (cal > 0),
                        "published": PUBLISHED.get((a, b)),
                        "same_arch": SAME_ARCH.get((a, b)),
                    })

    # Correlation between the pair calibration gap and the contrast distortion.
    def pearson(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if sx == 0 or sy == 0:
            return float("nan")
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        return pearson(rank(xs), rank(ys))

    stats = {}
    for tag, sub in [("all", rows),
                     ("max16", [r for r in rows if r["pooling"] == "max16"]),
                     ("extreme", [r for r in rows if r["threshold"] >= 181]),
                     ("published", [r for r in rows if r["published"]])]:
        if len(sub) < 3:
            continue
        stats[tag] = {
            "n": len(sub),
            "pearson": pearson([r["devgap"] for r in sub],
                               [r["distort"] for r in sub]),
            "spearman": spearman([r["devgap"] for r in sub],
                                 [r["distort"] for r in sub]),
            "flips": sum(1 for r in sub if r["flip"]),
            "max_abs_distort": max(abs(r["distort"]) for r in sub),
        }

    L = []
    W = L.append
    W("# Published contrasts before and after calibrating both sides\n")
    W("`raw` is the contrast a table reports; `cal` is the same contrast with one "
      "global monotone quantile map fitted per arm on the calibration half and held "
      "fixed across every cell. Neither knob sees the evaluation half and neither "
      "can reorder pixels.\n")

    W("\n## Does the calibration gap predict the distortion?\n")
    W("| subset | n | Pearson | Spearman | sign flips | max abs distortion |")
    W("|---|---|---|---|---|---|")
    for tag, s in stats.items():
        W(f"| {tag} | {s['n']} | {s['pearson']:+.3f} | {s['spearman']:+.3f} | "
          f"{s['flips']} | {s['max_abs_distort']:.4f} |")

    W("\n## Contrasts drawn from published comparisons, max16\n")
    W("| comparison | thr | bias A | bias B | raw | cal | rel raw | rel cal | flip |")
    W("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["published"] or r["pooling"] != "max16":
            continue
        rr = "n/a" if r["rel_raw"] is None else f"{100*r['rel_raw']:+.1f}%"
        rc = "n/a" if r["rel_cal"] is None else f"{100*r['rel_cal']:+.1f}%"
        W(f"| {ARMS[r['b']]} vs {ARMS[r['a']]} | {r['threshold']} | "
          f"{r['bias_a']:.3f} | {r['bias_b']:.3f} | {r['raw']:+.4f} | "
          f"{r['cal']:+.4f} | {rr} | {rc} | {'YES' if r['flip'] else ''} |")

    W("\n## One architecture, two trainings, max16\n")
    W("Not a published head-to-head. Both arms are the same cuboid transformer; "
      "one checkpoint is EarthFormer's release, the other is the CasCast authors' "
      "retraining of it as their deterministic backbone.\n")
    W("| thr | bias EF | bias CasCast-bb | raw | cal | rel raw | rel cal | flip |")
    W("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r.get("same_arch") or r["pooling"] != "max16":
            continue
        rr = "n/a" if r["rel_raw"] is None else f"{100*r['rel_raw']:+.1f}%"
        rc = "n/a" if r["rel_cal"] is None else f"{100*r['rel_cal']:+.1f}%"
        W(f"| {r['threshold']} | {r['bias_a']:.3f} | {r['bias_b']:.3f} | "
          f"{r['raw']:+.4f} | {r['cal']:+.4f} | {rr} | {rc} | "
          f"{'YES' if r['flip'] else ''} |")

    W("\n## Sources for the published pairs\n")
    for (a, b), src in PUBLISHED.items():
        W(f"- **{ARMS[b]} vs {ARMS[a]}**: {src}")

    W("\n## Every sign flip, any pooling\n")
    W("| comparison | pooling | thr | raw | cal |")
    W("|---|---|---|---|---|")
    for r in rows:
        if r["flip"]:
            W(f"| {ARMS[r['b']]} vs {ARMS[r['a']]} | {r['pooling']} | "
              f"{r['threshold']} | {r['raw']:+.4f} | {r['cal']:+.4f} |")

    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps(
        {"rows": rows, "stats": stats, "arms": ARMS,
         "published": {f"{a}|{b}": v for (a, b), v in PUBLISHED.items()}}, indent=1))
    print(f"wrote {args.out_md} and {args.out_json}: {len(rows)} contrast cells")
    for tag, s in stats.items():
        print(f"  {tag:10s} n={s['n']:4d} r={s['pearson']:+.3f} "
              f"rho={s['spearman']:+.3f} flips={s['flips']}")


if __name__ == "__main__":
    main()
