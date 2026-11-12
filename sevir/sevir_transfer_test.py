"""Out-of-sample test of the bias-gap relation on systems not used to fit it.

Predictor: deviation gap g = |log B_a| - |log B_b| (raw frequency biases, margin b - a).
Response:  distortion D = raw margin - calibrated margin (exact post-pooling threshold control).

Fit D = c + s*g on the 450 six-arm SEVIR contrasts (results/revision/postpool/sevir_postpool_contrasts.json),
then predict D for DiffCast (results/revision/diffcast) and PreDiff (results/revision/prediff) contrasts, which were
not used in the fit. Reports rank and sign transfer, coverage of the 95% prediction band inside and
outside the gap range seen in the fit, and the zero-gap behaviour.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402
RES = P.RESULTS / "revision"


def dev(b):
    return abs(math.log(b)) if b and b > 0 else None


def load_test(path):
    out = []
    for r in json.loads(Path(path).read_text())["contrasts"]:
        da, db = dev(r["bias_a"]), dev(r["bias_b"])
        if da is None or db is None or (r["raw"] == 0 and r["cal"] == 0):
            continue
        out.append({"pair": f"{r['b']} - {r['a']}", "pooling": r["pooling"], "threshold": r["threshold"],
                    "g": da - db, "D": r["raw"] - r["cal"]})
    return out


def main():
    S = json.loads((RES / "postpool" / "sevir_postpool_contrasts.json").read_text())["rows"]
    x = np.array([r["devgap"] for r in S])
    y = np.array([r["distort_post_exact"] for r in S])
    X = np.c_[np.ones_like(x), x]
    (c, s), *_ = np.linalg.lstsq(X, y, rcond=None)
    sd = math.sqrt(((y - X @ [c, s]) ** 2).sum() / (len(y) - 2))
    lo, hi = x.min(), x.max()
    z = np.abs(x) < 0.05
    L = ["# Out-of-sample test of the bias-gap relation", "",
         "g = |log B_a| − |log B_b|; D = raw − calibrated margin (exact post-pooling threshold).", "",
         f"**Fit on the six SEVIR arms** (450 contrasts): D = {c:+.4f} + {s:+.4f}·g, residual sd {sd:.4f}, "
         f"r {np.corrcoef(x, y)[0, 1]:.3f}, ρ {spearmanr(x, y).statistic:.3f}. Gap range in the fit "
         f"[{lo:+.2f}, {hi:+.2f}]. Cells with |g| < 0.05: n = {z.sum()}, mean |D| = {np.abs(y[z]).mean():.4f}.", "",
         "| held-out system | n | g range | ρ(g, D) | sign agreement, abs(g) > 0.1 | inside fitted range: n, within 95% band | outside range: n | mean abs error |",
         "|---|---|---|---|---|---|---|---|"]
    out = {"fit": {"intercept": c, "slope": s, "resid_sd": sd, "range": [lo, hi]}, "tests": {}}
    for lab, sub in [("DiffCast (CVPR 2024): DiffCast, PhyDNet backbone, persistence", "diffcast/diffcast_table.json"),
                     ("PreDiff (NeurIPS 2023) − persistence", "prediff/prediff_table.json")]:
        T = load_test(RES / sub)
        g = np.array([r["g"] for r in T]); D = np.array([r["D"] for r in T]); pr = c + s * g
        m = (g >= lo) & (g <= hi); big = np.abs(g) > 0.1
        inb = np.abs(D - pr) <= 1.96 * sd
        rho = spearmanr(g, D).statistic
        sgn = float(np.mean(np.sign(D[big]) == np.sign(g[big]))) if big.any() else float("nan")
        L.append(f"| {lab} | {len(T)} | [{g.min():+.2f}, {g.max():+.2f}] | {rho:.3f} | "
                 f"{sgn:.0%} (n = {big.sum()}) | {m.sum()}, {inb[m].mean():.0%} | {(~m).sum()} | {np.abs(D - pr).mean():.4f} |")
        out["tests"][lab] = {"n": len(T), "rho": rho, "sign_agree": sgn, "n_big": int(big.sum()),
                             "n_inside": int(m.sum()), "band_inside": float(inb[m].mean()),
                             "n_outside": int((~m).sum()), "mae": float(np.abs(D - pr).mean()),
                             "max_abs_g": float(np.abs(g).max()), "max_abs_D": float(np.abs(D).max())}
    L += ["", "The linear form is fitted on gaps up to about 1.7; outside that range CSI's bounds make D saturate, "
          "so only rank and sign are tested there."]
    (RES / "SEVIR_TRANSFER_TEST.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RES / "sevir_transfer_test.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
