"""Appendix L's intervals recomputed under the prior-period fit of Appendix J.

Appendix L answers "how much of the cascade's advantage survives when both arms
carry the same transform", with the transform fitted on the disjoint calibration
half of the test split. Appendix J shows the point estimate of that residual
falls when the transform is instead fitted on the earlier validation window, from
+12.9% to +4.6% at the headline cell. A point estimate without an interval cannot
say whether anything survives, and the honest answer matters in both directions:
if the interval excludes zero the cascade retains real skill under a
deployment-realistic fit; if it includes zero, the whole published gain is
indistinguishable from calibration once the map is fitted out of period.

This script answers it from the per-event count cache alone. No HDF5, no torch,
no GPU: reading and pooling the prediction fields is what costs half an hour, and
`contrast_bootstrap.py` already does that work once for every knob including this
one. What is left here is resampling event indices, which is numpy.

Protocol is deliberately identical to Appendix L rather than better: the same
2000 replicates, the same seed, and one event-index vector shared by every arm
within a replicate so the contrast stays paired. Raising the replicate count
would change nothing a reader cares about and would break the symmetry that lets
the two appendices be read side by side.

Usage
-----
    python sevir/valfit_bootstrap.py

Requires a cache carrying the `qmap_val` knob. If yours predates that column,
delete it and rerun contrast_bootstrap.py with the sevir_valpred_*.h5 caches and
the val subset in place; that is the one pass over the fields.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P                                            # noqa: E402

POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]
THRESHOLDS = [16, 74, 133, 160, 181, 219]
DET, CASC = "cascast_det", "cascast_cascade"


def csi(c):
    h, f, m = c[..., 0], c[..., 1], c[..., 2]
    return h / np.maximum(h + f + m, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_perevent_counts.npz"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooling", default="max16")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_valfit_bootstrap.json"))
    ap.add_argument("--out-md", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_valfit_bootstrap.md"))
    args = ap.parse_args()

    P.require(args.cache, "per-event count cache (run sevir/contrast_bootstrap.py)")
    z = np.load(args.cache, allow_pickle=True)
    counts, arms = z["counts"], [str(a) for a in z["arms"]]
    if "knobs" not in z.files:
        raise SystemExit(
            f"{args.cache} predates the knob-name column and cannot be indexed "
            f"by name. Delete it and rerun sevir/contrast_bootstrap.py.")
    knobs = [str(k) for k in z["knobs"]]
    for want in ("identity", "qmap_global", "qmap_val"):
        if want not in knobs:
            raise SystemExit(
                f"cache has knobs {knobs}, missing {want!r}. Rerun "
                f"sevir/contrast_bootstrap.py with the sevir_valpred_*.h5 "
                f"caches and the val subset present.")

    pi = POOLINGS.index(args.pooling)
    ai, ci = arms.index(DET), arms.index(CASC)
    k_id, k_test, k_val = (knobs.index("identity"), knobs.index("qmap_global"),
                           knobs.index("qmap_val"))

    n_ev = counts.shape[0]
    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0, n_ev, size=(args.n_boot, n_ev))   # shared -> paired

    rows = []
    for ti, t in enumerate(THRESHOLDS):
        cell = {"threshold": t, "pooling": args.pooling, "csi": {}, "contrasts": {}}
        for label, arm, kn in (("backbone", ai, k_id), ("cascade", ci, k_id),
                               ("backbone+test", ai, k_test),
                               ("cascade+test", ci, k_test),
                               ("backbone+val", ai, k_val),
                               ("cascade+val", ci, k_val)):
            cell["csi"][label] = float(csi(counts[:, arm, pi, ti, kn, :].sum(0)))

        def contrast(a_arm, a_kn, b_arm, b_kn):
            ca = counts[:, a_arm, pi, ti, a_kn, :]
            cb = counts[:, b_arm, pi, ti, b_kn, :]
            d = csi(np.stack([cb[b].sum(0) for b in boot])) - \
                csi(np.stack([ca[b].sum(0) for b in boot]))
            obs = float(csi(cb.sum(0)) - csi(ca.sum(0)))
            p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
            return {"diff": obs,
                    "ci95": [float(np.percentile(d, 2.5)),
                             float(np.percentile(d, 97.5))],
                    "p": min(p, 1.0), "excludes_zero": bool(
                        np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

        cell["contrasts"]["residual_test"] = contrast(ai, k_test, ci, k_test)
        cell["contrasts"]["residual_val"] = contrast(ai, k_val, ci, k_val)
        cell["contrasts"]["knob_value_val"] = contrast(ai, k_id, ai, k_val)
        rows.append(cell)

    args.out.write_text(json.dumps(
        {"n_boot": args.n_boot, "seed": args.seed, "n_eval": n_ev,
         "pooling": args.pooling, "table": rows}, indent=2))

    L = ["# The residual under the prior-period fit, with intervals\n",
         f"Paired event-level bootstrap, {args.n_boot} replicates over {n_ev} "
         f"evaluation events, {args.pooling} pooling. Same seed, replicate count "
         f"and shared event-index vectors as Appendix L, so the two are directly "
         f"comparable. `residual` is cascade minus backbone with BOTH arms "
         f"carrying the same transform.\n",
         "| tau | residual (test-fit) | residual (val-fit) | val-fit excludes 0 |",
         "|---|---|---|---|"]
    for r in rows:
        a, b = r["contrasts"]["residual_test"], r["contrasts"]["residual_val"]
        L.append(f"| {r['threshold']} | {a['diff']:+.4f} "
                 f"[{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | "
                 f"{b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
                 f"{'yes' if b['excludes_zero'] else 'NO'} |")
    n_excl = sum(r["contrasts"]["residual_val"]["excludes_zero"] for r in rows)
    L.append(f"\nThe val-fitted residual excludes zero at {n_excl} of "
             f"{len(rows)} thresholds.")
    args.out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out} and {args.out_md}")


if __name__ == "__main__":
    main()
