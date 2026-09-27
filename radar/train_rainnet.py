"""Train RainNet (single-decoder, plain MSE baseline) on RY radar data.

This is the single-decoder comparator row for the thesis test. Uses the
same data + optimizer + random seed protocol as `train_rainnet_dual_decoder.py`
so that paired-t tests across seeds are honest.

Loss: plain nn.MSELoss on the log-transformed target.
(The `--loss logcosh` option is available for reproducing the original
RainNet paper but is NOT the thesis baseline. It introduces a loss
confound vs the DualDecoder comparison. Use the default MSE.)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
# Dataset and model definitions are vendored under common/ so this runs from a
# clean checkout.
import sys as _s2
from pathlib import Path as _P2
_C = _P2(__file__).resolve().parents[1] / "common"
_s2.path[:0] = [str(_C), str(_C / "data"), str(_C / "models")]
from data.dataset_rainnet import RainNetRYDataset, LOG_OFFSET
from models.rainnet import RainNet

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        d = pred - target
        return torch.mean(d + torch.nn.functional.softplus(-2.0 * d) - math.log(2.0))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", type=Path, default=Path(str(P.RADAR_H5)))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--loss", choices=["mse", "logcosh"], default="mse")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--gradient-clip", type=float, default=10.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=None,
                   help="RNG seed for random subsampling (see train_rainnet_dual_decoder.py).")
    p.add_argument("--critical-threshold-mm", type=float, default=0.5,
                   help="Eval CSI threshold in raw mm/5min (not used in loss, only val metric)")
    p.add_argument("--split-regime", choices=["default", "paper"], default="default")
    p.add_argument("--summer-only", action="store_true")
    p.add_argument("--wet-filter", action="store_true")
    p.add_argument("--wet-min-frac", type=float, default=0.10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mae = args.output_dir / "rainnet_best_mae.pth"
    ckpt_csi = args.output_dir / "rainnet_best_csi.pth"

    ds_kwargs = dict(split_regime=args.split_regime, summer_only=args.summer_only,
                     wet_filter=args.wet_filter, wet_min_frac=args.wet_min_frac)
    train_ds = RainNetRYDataset(args.hdf5, split="train",
                                max_samples=args.max_train_samples,
                                subset_seed=args.subset_seed, **ds_kwargs)
    val_ds = RainNetRYDataset(args.hdf5, split="val",
                              max_samples=args.max_val_samples,
                              subset_seed=args.subset_seed, **ds_kwargs)
    print(f"train: {len(train_ds):,}  val: {len(val_ds):,}  "
          f"(regime={args.split_regime}, summer={args.summer_only}, wet={args.wet_filter})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    (args.output_dir / "train_config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model": "RainNet", "device": str(device),
    }, indent=2))

    model = RainNet(in_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() if args.loss == "mse" else LogCoshLoss()
    # Use bfloat16 autocast (fp32 dynamic range, no GradScaler / no overflow risk).
    use_amp = args.amp and device.type == "cuda"

    best_mae = math.inf
    best_csi = -1.0
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        sums = 0.0
        nb = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        for predictor, target_log, _ in pbar:
            predictor = predictor.to(device, non_blocking=True)
            target_log = target_log.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred_log = model(predictor)
                loss = criterion(pred_log, target_log)

            if not torch.isfinite(loss):
                print(f"   [skip] non-finite loss ({loss.item()}), batch skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            sums += loss.item()
            nb += 1
            pbar.set_postfix({"loss": f"{sums/max(nb,1):.4f}"})

        model.eval()
        ae = sse = npix = 0.0
        hits = misses = fa = 0
        with torch.no_grad():
            for predictor, target_log, _ in val_loader:
                predictor = predictor.to(device, non_blocking=True)
                target_log = target_log.to(device, non_blocking=True)
                pred_log = model(predictor)
                pred_mm = torch.exp(pred_log) - LOG_OFFSET
                tgt_mm = torch.exp(target_log) - LOG_OFFSET
                diff = pred_mm - tgt_mm
                ae += diff.abs().sum().item()
                sse += (diff ** 2).sum().item()
                npix += pred_mm.numel()
                ph = pred_mm >= args.critical_threshold_mm
                gh = tgt_mm >= args.critical_threshold_mm
                hits += (ph & gh).sum().item()
                misses += (~ph & gh).sum().item()
                fa += (ph & ~gh).sum().item()
        val_mae = ae / max(npix, 1)
        denom = hits + misses + fa
        val_csi = hits / denom if denom > 0 else 0.0
        print(f"Epoch {epoch+1} | train loss {sums/nb:.4f} | val MAE {val_mae:.4f} mm  CSI {val_csi:.4f}")

        improved = False
        if val_mae < best_mae:
            best_mae = val_mae; torch.save(model.state_dict(), ckpt_mae); improved = True
            print(f"   best-MAE saved -> {ckpt_mae.name}")
        if val_csi > best_csi:
            best_csi = val_csi; torch.save(model.state_dict(), ckpt_csi); improved = True
            print(f"   best-CSI saved -> {ckpt_csi.name}")
        patience = 0 if improved else patience + 1
        if patience >= args.early_stopping_patience:
            print(f"   early stop after {patience} epochs without improvement")
            break

    print(f"\n-- done. best MAE={best_mae:.4f} mm  best CSI={best_csi:.4f} --")


if __name__ == "__main__":
    main()
