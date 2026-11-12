"""The exact control for crowd counting: fit the calibration map after patch summation.

In crowd/pooling_control.py the knob acts on pixels before the 32x32
patch sum (and before any further pooling of the patch grid). A nonlinear
member (the quantile map, or a clipping negative shift) can therefore reorder
patches. Here the map is fitted instead on the pooled block values of the
validation split, and applied to the pooled forecast. A monotone map applied
after pooling cannot reorder blocks, so it is exactly a per-arm threshold.

The map is the exact empirical quantile map from the forecast's pooled values
onto the observed pooled values, fitted on val: at each threshold tau the
forecast threshold is the one whose val exceedance count equals the observed
count. That is one monotone map per (arm, pooling), evaluated at the seven
thresholds. It is not a per-cell selection.

Raw scores are recomputed with the released functions and checked against
results/crowd_pooling_s<seed>.json before anything else is reported.

    CROWD_DATA=/path/to/ShanghaiTech CROWD_OUTPUTS=/path/to/outputs \
        python crowd/crowd_pool_then_calibrate.py --seeds 42,43,44,45,46
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import paths as P  # noqa: E402
import pooling_control as pcm  # noqa: E402  (released code, reused verbatim)

ARMS = ("baseline", "csrnet")


def load_model(arm, seed, outputs):
    ck = outputs / f"{arm}_s{seed}" / "best_model.pth"
    model = pcm.build(arm)
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    return model


def predict(model, loader, device):
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for img, _ in loader:
            p = model(img.to(device))
            out.append(np.maximum(p.squeeze().cpu().numpy(), 0).astype(np.float64))
    return out


def pooled_values(grids, how, k):
    vals = [pcm.pool(g, how, k) for g in grids]
    return [v for v in vals if v is not None]


def exact_thresholds(pred_val, gt_val):
    """Forecast threshold per tau so that val exceedance counts match."""
    p = np.sort(np.concatenate([v.ravel() for v in pred_val]))[::-1]
    g = np.concatenate([v.ravel() for v in gt_val])
    thr = {}
    for tau in pcm.THRESHOLDS:
        k = int(np.sum(g > tau))
        # events are value > threshold; with k targets, put the threshold at
        # the (k+1)-th largest forecast so exactly k forecast blocks exceed it
        thr[tau] = float(p[k]) if 0 < k < p.size else (float("inf") if k == 0 else -1.0)
    return thr


def score(pred, gt, t_pred, tau):
    H = F = M = 0
    for pp, gp in zip(pred, gt):
        pe, ge = pp > t_pred, gp > tau
        H += int(np.sum(pe & ge)); F += int(np.sum(pe & ~ge)); M += int(np.sum(~pe & ge))
    d = H + F + M
    return (H / d if d else float("nan"), (H + F) / (H + M) if H + M else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("CROWD_DATA", "")))
    ap.add_argument("--outputs", type=Path, default=Path(os.environ.get("CROWD_OUTPUTS", str(P.CROWD / "outputs"))))
    ap.add_argument("--out", type=Path, default=P.RESULTS / "revision" / "crowd_postpool.json")
    a = ap.parse_args()
    pcm.config.DATA_ROOT = a.data
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = {}
    for name, kw in (("val", dict(split="val", use_train_val_split=True)),
                     ("test", dict(split="test_data"))):
        ds = pcm.ShanghaiTechDataset(a.data, pcm.config.PART, augment=False, **kw)
        loaders[name] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    gts = {s: pcm.raw_fields(loaders[s]) for s in ("val", "test")}
    gtg = {s: [pcm.patch_grid(x) for x in gts[s]] for s in ("val", "test")}

    rows, worst = [], 0.0
    for seed in a.seeds.split(","):
        released = json.loads((P.RESULTS /
                               f"crowd_pooling_s{seed}.json").read_text())["table"]
        released = {(r["arm"], r["pooling"], r["threshold"]): r for r in released}
        for arm in ARMS:
            model = load_model(arm, seed, a.outputs)
            pr = {s: [pcm.patch_grid(x) for x in predict(model, loaders[s], dev)]
                  for s in ("val", "test")}
            for s in ("val", "test"):
                keep = [i for i, g in enumerate(gtg[s]) if g is not None and pr[s][i] is not None]
                pr[s] = [pr[s][i] for i in keep]
                gtg_s = [gtg[s][i] for i in keep]
                if s == "val":
                    gv = gtg_s
                else:
                    gt_test = gtg_s
            for lbl, how, k in pcm.POOLINGS:
                pv, gvv = pooled_values(pr["val"], how, k), pooled_values(gv, how, k)
                pt, gtt = pooled_values(pr["test"], how, k), pooled_values(gt_test, how, k)
                thr = exact_thresholds(pv, gvv)
                for tau in pcm.THRESHOLDS:
                    u = pcm.counts(pr["test"], gt_test, how, k, tau)
                    c_post, b_post = score(pt, gtt, thr[tau], tau)
                    ref = released[(arm, lbl, tau)]
                    if math.isfinite(u["csi"]) and math.isfinite(ref["csi_uncal"]):
                        worst = max(worst, abs(u["csi"] - ref["csi_uncal"]))
                    npos = int(round(u["R"] * u["n_blocks"]))
                    rows.append({"seed": seed, "arm": arm, "pooling": lbl, "threshold": tau,
                                 "csi_uncal": u["csi"], "bias_uncal": u["bias"],
                                 "csi_post": c_post, "bias_post": b_post,
                                 "csi_uncal_released": ref["csi_uncal"],
                                 "bias_uncal_released": ref["bias_uncal"],
                                 "usable_released": ref["usable"],
                                 "csi_knob_released": ref["csi_knob"],
                                 "csi_percell_released": ref["csi_percell"],
                                 "knob_percell_released": ref["knob_percell"],
                                 "n_pos": npos, "usable": npos >= 30})
            print(f"  seed {seed} {arm}: done", flush=True)

    # Released statistics use released raw scores only; the post-pooling
    # statistic uses this run's raw scores only. The two are never mixed.
    use = [r for r in rows if r["usable"] and all(
        math.isfinite(r[x]) for x in ("bias_uncal", "csi_uncal", "csi_post"))]
    rel = [r for r in rows if r["usable_released"] and all(
        math.isfinite(r[x]) for x in ("bias_uncal_released", "csi_uncal_released",
                                      "csi_percell_released", "csi_knob_released"))]

    def rho(rs, key, base="csi_uncal", bias="bias_uncal"):
        return -spearmanr([r[bias] for r in rs],
                          [r[key] - r[base] for r in rs]).statistic

    def rho_rel(rs, key):
        return rho(rs, key, "csi_uncal_released", "bias_uncal_released")

    none = [r for r in use if r["pooling"] == "none"]
    rel_none = [r for r in rel if r["pooling"] == "none"]
    summary = {
        "max_abs_raw_csi_vs_released": worst,
        "n_usable": len(use),
        "rho_postpool": rho(use, "csi_post"),
        "n_usable_released": len(rel),
        "rho_released_percell": rho_rel(rel, "csi_percell_released"),
        "rho_released_global": rho_rel(rel, "csi_knob_released"),
        "n_none": len(none),
        "rho_postpool_none": rho(none, "csi_post"),
        "n_none_released": len(rel_none),
        "rho_released_percell_none": rho_rel(rel_none, "csi_percell_released"),
    }
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for k_, v in summary.items():
        print(f"{k_}: {v:.4g}" if isinstance(v, float) else f"{k_}: {v}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
