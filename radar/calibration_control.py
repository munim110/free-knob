"""Post-hoc calibration control on radar: can a monotone knob rescue tail CSI?

Why radar is the decisive domain
--------------------------------
On atmosphere the single decoder's failure is one of *calibration*: it detects
convection but over-predicts its extent four- to six-fold. A monotone
recalibration is well suited to that, and measurably closes the gap.

On radar the failure is different in kind. The pretrained RainNet's CSI at 2.5
and 5 mm/5min is *exactly zero* -- not poorly calibrated but absent, zero hits
across 44,233 sequences. Rescuing it requires putting heavy rain somewhere, and a
monotone pointwise knob can certainly do that: scaling a field up, or remapping
its distribution onto the target's, manufactures pixels above 2.5 mm. What a
pointwise knob cannot do is decide *where*. So this experiment separates the two
capabilities the paper conflates under "detection": manufacturing tail intensity,
which is post-hoc, and localising it, which the paper claims needs the extreme
decoder.

If a scaled or quantile-mapped single decoder recovers the tail CSI, DualDecoder's
headline radar result is an amplitude effect. If it manufactures tail pixels in
the wrong places, giving high Bias and near-zero CSI, then the localisation claim is
established on the domain where it matters most.

Protocol
--------
The quantile map is fitted on the split the models actually validated on (2015,
May-Sep, wet-filtered: `split_regime='paper', split='val'`) and applied to the
published test split (2017 full year, unfiltered, `split_regime='default'`,
n=44,233, the same split and count as the paper's tables).

Because that val distribution is summer/wet-filtered while test is full-year, a
map fitted on it is distributionally handicapped. To keep the baseline steelmanned
rather than strawmanned, a second map is fitted on unfiltered full-year 2015 and
both are reported; the baseline is credited with whichever does better.

Everything is streamed: histograms for the fit, contingency counts for the
evaluation. No field is held in memory.

Knobs (single decoders)
    gain   mm -> g * mm, a pure multiplicative bias correction. Monotone and
           exactly zero-preserving, so it cannot invent drizzle.
    qmap   CDF matching in the model's native log space, fitted as above.
    qmap+gain  the composition, the strongest recipe available post hoc.

DualDecoder is swept over beta, giving its own frontier under the identical metric.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Dataset and model definitions are vendored under common/ so this runs from a
# clean checkout.
import sys as _s2
from pathlib import Path as _P2
_C = _P2(__file__).resolve().parents[1] / "common"
_s2.path[:0] = [str(_C), str(_C / "data"), str(_C / "models")]
from data.dataset_rainnet import RainNetRYDataset, LOG_OFFSET, PAD  # noqa: E402
from models.rainnet import RainNet  # noqa: E402
from models.rainnet_dual_decoder import RainNetDualDecoder  # noqa: E402
from models.rainnet_dualhead import RainNetDualHead  # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

THRESHOLDS = [0.1, 0.5, 1.0, 2.5, 5.0]
# trimmed for the full-test run: the smoke test showed gain alone and qmap alone
# each rescue the tail, so the compositions add cost without adding an answer.
GAINS = [1.0, 1.5, 2.0, 3.0, 4.0]
BETAS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
TAIL_LEVELS = [2.5, 5.0]

# histogram grid in the model's native log space: log(mm + LOG_OFFSET).
# mm=0 maps to log(0.01) = -4.605, so the dry atom occupies the first bin.
LOG_LO, LOG_HI, NBINS = math.log(LOG_OFFSET), math.log(200.0 + LOG_OFFSET), 4000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["rainnet", "dual_decoder", "dualhead"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hdf5", type=Path, default=Path(str(P.RADAR_H5)))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--fit-max-samples", type=int, default=3000,
                   help="cap on val sequences used to fit the quantile map")
    p.add_argument("--test-max-samples", type=int, default=None)
    p.add_argument("--beta", type=float, default=2.0,
                   help="beta used for a dual_decoder/dualhead checkpoint when fitting the map")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_model(kind, ckpt, device):
    state = torch.load(ckpt, map_location=device)
    if kind == "dual_decoder":
        model = RainNetDualDecoder(in_channels=4).to(device)
    elif kind == "dualhead":
        ch = tuple(state[f"dec{i}.block.0.weight"].shape[0] for i in (4, 3, 2, 1))
        model = RainNetDualHead(in_channels=4, dec_channels=ch).to(device)
    else:
        model = RainNet(in_channels=4).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def forward_log(model, kind, x, beta):
    """Return the model's prediction in log space (combined, for dual decoders)."""
    if kind in ("dual_decoder", "dualhead"):
        bg, ext = model(x)
        return bg + beta * ext, (bg, ext)
    return model(x), None


def center(t):
    return t[..., PAD:-PAD, PAD:-PAD]


# ----------------------------------------------------------------- fitting ---
def fit_histograms(model, kind, loader, device, beta, desc):
    """Accumulate log-space histograms of prediction and target over a split."""
    edges = torch.linspace(LOG_LO, LOG_HI, NBINS + 1, device=device)
    h_pred = torch.zeros(NBINS + 2, dtype=torch.float64, device=device)
    h_tgt = torch.zeros(NBINS + 2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for predictor, target_log, _ in tqdm(loader, desc=desc):
            predictor = predictor.to(device, non_blocking=True)
            target_log = target_log.to(device, non_blocking=True)
            pred_log, _ = forward_log(model, kind, predictor, beta)
            for src, h in ((center(pred_log), h_pred), (center(target_log), h_tgt)):
                idx = torch.bucketize(src.reshape(-1).float(), edges)
                h += torch.bincount(idx, minlength=NBINS + 2).to(torch.float64)
    return edges, h_pred, h_tgt


def build_qmap(edges, h_pred, h_tgt):
    """Monotone CDF-matching map on the log-space bin grid.

    Returns (src_edges, dst_values): bucketize a log prediction into src_edges and
    index dst_values to obtain its mapped log value. Piecewise constant at the bin
    resolution (~0.0025 log units over 4000 bins), which is far finer than any
    threshold of interest.
    """
    # bin representative values: use upper edge for interior bins
    centers = torch.cat([edges[:1], edges])            # len NBINS+2
    cdf_p = torch.cumsum(h_pred, 0) / h_pred.sum()
    cdf_t = torch.cumsum(h_tgt, 0) / h_tgt.sum()
    # for each prediction bin, find the target value at the same cumulative prob
    cdf_p_np, cdf_t_np, cen_np = (cdf_p.cpu().numpy(), cdf_t.cpu().numpy(),
                                  centers.cpu().numpy())
    # np.interp needs increasing x; cdf_t is non-decreasing. Deduplicate.
    keep = np.concatenate([[True], np.diff(cdf_t_np) > 0])
    dst = np.interp(cdf_p_np, cdf_t_np[keep], cen_np[keep])
    dst = np.maximum.accumulate(dst)                    # enforce monotonicity
    return edges, torch.tensor(dst, dtype=torch.float32, device=edges.device)


def apply_qmap(pred_log, edges, dst_values):
    idx = torch.bucketize(pred_log, edges)
    return dst_values[idx]


# -------------------------------------------------------------- evaluation ---
class Accum:
    """Streaming contingency counts and tail-intensity sums for one arm/knob."""

    def __init__(self):
        self.c = {t: [0, 0, 0, 0] for t in THRESHOLDS}
        self.ae = 0.0
        self.npix = 0.0
        self.tail = {t: [0.0, 0.0, 0.0] for t in TAIL_LEVELS}  # sum|e|, sum pred, n

    def update(self, pred_mm, tgt_mm):
        d = (pred_mm - tgt_mm).abs()
        self.ae += float(d.sum()); self.npix += pred_mm.numel()
        for t in THRESHOLDS:
            ph, gh = pred_mm >= t, tgt_mm >= t
            self.c[t][0] += int((ph & gh).sum())
            self.c[t][1] += int((~ph & gh).sum())
            self.c[t][2] += int((ph & ~gh).sum())
            self.c[t][3] += int((~ph & ~gh).sum())
        for t in TAIL_LEVELS:
            m = tgt_mm >= t
            n = int(m.sum())
            if n:
                self.tail[t][0] += float(d[m].sum())
                self.tail[t][1] += float(pred_mm[m].sum())
                self.tail[t][2] += n

    def result(self):
        out = {"mae_mm": self.ae / max(self.npix, 1), "per_threshold": {}}
        for t in THRESHOLDS:
            h, m, f, cn = self.c[t]
            out["per_threshold"][f"{t:g}mm"] = {
                "csi": h / (h + m + f) if (h + m + f) else 0.0,
                "pod": h / (h + m) if (h + m) else 0.0,
                "far": f / (h + f) if (h + f) else 0.0,
                "bias": (h + f) / (h + m) if (h + m) else 0.0,
                "hits": h, "misses": m, "false_alarms": f,
            }
        for t in TAIL_LEVELS:
            s, sp, n = self.tail[t]
            out[f"tail_{t:g}mm"] = {
                "mae_on_true_tail_mm": s / n if n else float("nan"),
                "mean_pred_on_true_tail_mm": sp / n if n else float("nan"),
                "n_true_tail_pixels": int(n),
            }
        return out


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.model, args.checkpoint, device)

    # --- fit the two quantile maps -------------------------------------------
    maps = {}
    for name, kw in (("qmap_paperval",
                      dict(split="val", split_regime="paper", summer_only=True,
                           wet_filter=True, wet_min_frac=0.10)),
                     ("qmap_2015full",
                      dict(split="val", split_regime="paper", summer_only=False,
                           wet_filter=False))):
        ds = RainNetRYDataset(args.hdf5, max_samples=args.fit_max_samples, **kw)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        print(f"fit {name}: {len(ds):,} sequences")
        edges, hp, ht = fit_histograms(model, args.model, ld, device, args.beta,
                                       f"fit {name}")
        maps[name] = build_qmap(edges, hp, ht)
        del ds, ld

    # --- evaluate on the published test split --------------------------------
    test = RainNetRYDataset(args.hdf5, split="test", max_samples=args.test_max_samples)
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"test: {len(test):,} sequences (default regime = 2017)")

    # one accumulator per (knob, value)
    accs = {}
    if args.model == "rainnet":
        for g in GAINS:
            accs[f"gain={g}"] = Accum()
        for mname in maps:
            accs[f"{mname}"] = Accum()
    else:
        # beta frontier, kept for reference: beta is an architectural blend
        # weight, and therefore outside the post-hoc knob family.
        for b in BETAS:
            accs[f"beta={b}"] = Accum()
        # matched control: the dual decoder held at its published beta, then
        # given exactly the knobs the single decoder receives. Without this the
        # comparison calibrates one side only.
        for g in GAINS:
            accs[f"gain={g}"] = Accum()
        for mname in maps:
            accs[f"{mname}"] = Accum()

    with torch.no_grad():
        for predictor, target_log, _ in tqdm(loader, desc=f"eval {args.model}"):
            predictor = predictor.to(device, non_blocking=True)
            target_log = target_log.to(device, non_blocking=True)
            tgt_mm = center(torch.exp(target_log) - LOG_OFFSET)

            if args.model == "rainnet":
                pred_log = center(model(predictor))
                base_mm = torch.exp(pred_log) - LOG_OFFSET
                for g in GAINS:
                    accs[f"gain={g}"].update(base_mm * g, tgt_mm)
                for mname, (edges, dst) in maps.items():
                    mapped_mm = torch.exp(apply_qmap(pred_log, edges, dst)) - LOG_OFFSET
                    accs[mname].update(mapped_mm, tgt_mm)
            else:
                bg, ext = model(predictor)
                bg, ext = center(bg), center(ext)
                for b in BETAS:
                    accs[f"beta={b}"].update(
                        torch.exp(bg + b * ext) - LOG_OFFSET, tgt_mm)
                # matched knobs on the combined field at the published beta
                comb_log = bg + args.beta * ext
                base_mm = torch.exp(comb_log) - LOG_OFFSET
                for g in GAINS:
                    accs[f"gain={g}"].update(base_mm * g, tgt_mm)
                for mname, (edges, dst) in maps.items():
                    mapped_mm = torch.exp(
                        apply_qmap(comb_log, edges, dst)) - LOG_OFFSET
                    accs[mname].update(mapped_mm, tgt_mm)

    out = {"model": args.model, "checkpoint": str(args.checkpoint),
           "n_test": len(test), "thresholds": THRESHOLDS,
           "arms": {k: a.result() for k, a in accs.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    for k, r in out["arms"].items():
        pt = r["per_threshold"]
        print(f"  {k:26s} CSI@0.5={pt['0.5mm']['csi']:.4f} "
              f"CSI@2.5={pt['2.5mm']['csi']:.4f} CSI@5={pt['5mm']['csi']:.4f} "
              f"Bias@2.5={pt['2.5mm']['bias']:.3f}")


if __name__ == "__main__":
    main()
