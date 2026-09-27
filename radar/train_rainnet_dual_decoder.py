"""Train RainNet-DualDecoder (dual-decoder unified decomposition) on RY radar data.

Mirrors src/train_dual_decoder.py's three-term loss and dual-checkpoint protocol,
adapted for high-value critical (heavy rain) and log(x+0.01) target space.

Loss:
    L = bg_w * MSE(bg_pred, target_log)                              # full-field
      + ext_w * ImportanceWeightedMSE(ext_pred, ext_target_log)      # critical residual
      + combined_w * MSE(bg_pred + beta * ext_pred, target_log)      # coupled output

    ext_target_log = clamp(target_log - tau_log, 0)    # non-negative excess

Checkpoints per run:
    <prefix>_best_mae.pth   lowest val MAE (mm/5min) on the combined prediction
    <prefix>_best_csi.pth   highest pixel-wise val CSI at tau on combined pred

Defaults reproduce the DualDecoder paper protocol (bg=ext=cmb=1.0, lambda=10, beta=2.0).
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
from data.dataset_rainnet import RainNetRYDataset, mm_to_log, log_to_mm, LOG_OFFSET
from losses.decomposition import (ImportanceWeightedMSELoss, SignedImportanceWeightedMSELoss,
                                  decompose_frequency, decompose_unified)
from models.rainnet_dual_decoder import RainNetDualDecoder
from models.rainnet_dualhead import RainNetDualHead

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", type=Path, default=Path(str(P.RADAR_H5)))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--gradient-clip", type=float, default=10.0)
    p.add_argument("--warmup-epochs", type=float, default=0.0,
                   help="Linear LR warmup over the first N epochs (0 = off, the "
                        "historical radar recipe). The frequency decomposition "
                        "starts with a near-zero-gradient extreme head, which is "
                        "the fragile stage: without warmup it collapsed on radar "
                        "seed 43 (epoch-1 bg loss 3.19 vs 0.08 on seed 42), the "
                        "same artifact documented for frequency-DualDecoder on crowd.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=None,
                   help="RNG seed for random subsampling of train/val. If set, "
                        "pick max-*-samples uniformly at random from the full "
                        "year range (instead of the first N chronologically).")

    # Architecture: dual-decoder DualDecoder, or shared-decoder dual-head (ablation)
    p.add_argument("--dec-width", type=float, default=1.0,
                   help="dualhead only: width multiplier on the single shared decoder. "
                        "DualDecoder is this net with the decoder duplicated (same encoder, 2x the "
                        "decoder), so at 1.0 the DualDecoder-vs-dualhead gap confounds decoder COUNT "
                        "with 40%% more parameters. 1.622 makes the two nets equal size "
                        "(43.89M vs 43.92M), isolating the count.")
    p.add_argument("--arch", choices=["dual_decoder", "dualhead"], default="dual_decoder",
                   help="dual_decoder = two independent decoders (RainNetDualDecoder, 43.92M params); "
                        "dualhead = single shared decoder + two output heads (RainNetDualHead, "
                        "31.38M params). Same training loss, same API. Used together to "
                        "separate 'two decoders' from 'two outputs'.")

    # Decomposition
    p.add_argument("--critical-threshold-mm", type=float, default=0.5,
                   help="tau in raw mm/5min. Central of the sensitivity sweep "
                        "{0.25, 0.5, 0.75, 1.0}; ~0.19%% of pixels at 0.5.")
    p.add_argument("--decomposition", choices=["unified", "frequency"], default="unified",
                   help="unified: ext=relu(y-tau_log), a pointwise function of y, so the ext "
                        "head re-solves the bg head's problem on the tail. frequency: "
                        "ext=y-G_sigma*y (signed), height above the LOCAL background -- "
                        "neighbourhood-dependent, so a different task. Frequency "
                        "REQUIRES --ext-activation linear (the residual is signed).")
    p.add_argument("--decomp-sigma", type=float, default=10.0,
                   help="Gaussian sigma in pixels for --decomposition frequency.")
    p.add_argument("--ext-activation", choices=["relu", "linear"], default="relu",
                   help="relu suits the unified decomposition's non-negative excess. The "
                        "crowd DualDecoder, where frequency decomposition works, uses a LINEAR head.")
    p.add_argument("--critical-mode", choices=["low", "high"], default="high",
                   help="Precipitation is high-value critical (heavy rain = extreme)")
    p.add_argument("--beta", type=float, default=2.0,
                   help="Training-time combiner: y_hat = bg + sign*beta*ext")
    p.add_argument("--bg-weight", type=float, default=1.0)
    p.add_argument("--ext-weight", type=float, default=1.0)
    p.add_argument("--combined-weight", type=float, default=1.0)
    p.add_argument("--lambda-weight", type=float, default=100.0,
                   help="Ext-head importance weight: w = 1 + lambda * ext_target. "
                        "Raised from DualDecoder's TBB default of 10 because log-precip "
                        "residuals are ~10x smaller than Kelvin residuals, and "
                        "critical pixels are ~15x sparser. lambda=100 restores "
                        "TBB-regime critical-pixel emphasis.")

    # Data splits / filters (paper regime: 2006-2013 train, 2015 val, 2016-17 test,
    # May-Sep only, >=10% wet-domain filter at forecast time)
    p.add_argument("--split-regime", choices=["default", "paper"], default="default")
    p.add_argument("--summer-only", action="store_true", help="May-Sep filter (paper).")
    p.add_argument("--wet-filter", action="store_true",
                   help="Drop sequences with <10%% of domain wet at forecast time (paper).")
    p.add_argument("--wet-min-frac", type=float, default=0.10)

    # Per-head loss choice (Option A = bg:logcosh, ext:iw-mse, cmb:mse)
    p.add_argument("--loss-bg", choices=["mse", "logcosh"], default="mse")
    p.add_argument("--loss-cmb", choices=["mse", "logcosh"], default="mse")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


class LogCoshLoss(nn.Module):
    """L = mean(log(cosh(pred - target))), numerically stable softplus form."""

    def forward(self, pred, target):
        e = pred - target
        # softplus(-2x) + x - log(2) == log(cosh(x)) without cosh overflow
        return (torch.nn.functional.softplus(-2.0 * e.abs()) + e.abs() - math.log(2.0)).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint filenames stay keyed by the legacy "rainnet_dual_decoder" prefix so the
    # existing eval / aggregator code works for both architectures.
    ckpt_mae = args.output_dir / "rainnet_dual_decoder_best_mae.pth"
    ckpt_csi = args.output_dir / "rainnet_dual_decoder_best_csi.pth"

    # tau in the log target space: the decomposition lives where the model outputs live
    tau_log = mm_to_log(args.critical_threshold_mm)
    sign = +1.0 if args.critical_mode == "high" else -1.0

    # Data
    ds_kwargs = dict(
        split_regime=args.split_regime,
        summer_only=args.summer_only,
        wet_filter=args.wet_filter,
        wet_min_frac=args.wet_min_frac,
    )
    train_ds = RainNetRYDataset(args.hdf5, split="train",
                                max_samples=args.max_train_samples,
                                subset_seed=args.subset_seed,
                                **ds_kwargs)
    val_ds = RainNetRYDataset(args.hdf5, split="val",
                              max_samples=args.max_val_samples,
                              subset_seed=args.subset_seed,
                              **ds_kwargs)
    print(f"train: {len(train_ds):,}  val: {len(val_ds):,}  "
          f"(regime={args.split_regime}, summer={args.summer_only}, wet={args.wet_filter})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    # Persist config for eval
    (args.output_dir / "train_config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "tau_log": float(tau_log),
        "sign": sign,
        "warmup_epochs": args.warmup_epochs, "model": "RainNetDualDecoder",
        "device": str(device),
    }, indent=2))

    # Model, optim, loss
    if args.decomposition == "frequency" and args.ext_activation != "linear":
        raise SystemExit("--decomposition frequency needs --ext-activation linear: the "
                         "residual y - G_sigma*y is signed and a ReLU head cannot "
                         "represent its negative part.")
    ModelCls = RainNetDualDecoder if args.arch == "dual_decoder" else RainNetDualHead
    model = (ModelCls(in_channels=4, ext_activation=args.ext_activation)
             if args.arch == "dual_decoder"
             else ModelCls(in_channels=4, dec_width=args.dec_width)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch}  params={n_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    warmup_steps = int(args.warmup_epochs * len(train_loader))
    if warmup_steps > 0:
        print(f"LR warmup: linear over {args.warmup_epochs:g} epochs "
              f"({warmup_steps:,} steps)")
    def _set_lr(step):
        if warmup_steps <= 0:
            return
        if step < warmup_steps:
            f = 1e-3 + (1.0 - 1e-3) * (step / warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = args.learning_rate * f
        elif step == warmup_steps:
            for g in optimizer.param_groups:
                g["lr"] = args.learning_rate
    global_step = 0
    # Use bfloat16 autocast: it has fp32's dynamic range, so no GradScaler and no overflow risk.
    # On 928x928 (~862k pixels) with MSE, fp16 grads overflow and corrupt weights.
    use_amp = args.amp and device.type == "cuda"

    def _make_loss(name):
        return LogCoshLoss() if name == "logcosh" else nn.MSELoss()

    bg_crit = _make_loss(args.loss_bg)
    ext_crit = (SignedImportanceWeightedMSELoss(lambda_weight=args.lambda_weight,
                                                mode=args.critical_mode)
                if args.decomposition == "frequency"
                else ImportanceWeightedMSELoss(lambda_weight=args.lambda_weight))
    cmb_crit = _make_loss(args.loss_cmb)
    print(f"loss: bg={args.loss_bg}  ext=iw-mse(lambda={args.lambda_weight})  cmb={args.loss_cmb}")

    best_mae = math.inf
    best_csi = -1.0
    patience = 0

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        sums = dict(loss=0.0, bg=0.0, ext=0.0, cmb=0.0)
        nb = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train beta={args.beta}]")
        for predictor, target_log, _target_mm in pbar:
            predictor = predictor.to(device, non_blocking=True)
            target_log = target_log.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                bg_pred, ext_pred = model(predictor)
                if args.decomposition == "frequency":
                    _, ext_target_log = decompose_frequency(target_log, args.decomp_sigma)
                else:
                    _, ext_target_log = decompose_unified(target_log, tau_log,
                                                          mode=args.critical_mode)
                combined_pred = bg_pred + sign * args.beta * ext_pred

                l_bg = bg_crit(bg_pred, target_log)
                l_ext = ext_crit(ext_pred, ext_target_log)
                l_cmb = cmb_crit(combined_pred, target_log)
                loss = args.bg_weight * l_bg + args.ext_weight * l_ext + args.combined_weight * l_cmb

            if not torch.isfinite(loss):
                print(f"   [skip] non-finite loss ({loss.item()}), batch skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            _set_lr(global_step)
            optimizer.step()
            global_step += 1

            sums["loss"] += loss.item(); sums["bg"] += l_bg.item()
            sums["ext"] += l_ext.item(); sums["cmb"] += l_cmb.item()
            nb += 1
            pbar.set_postfix({k: f"{v/max(nb,1):.4f}" for k, v in sums.items()})

        # ---- val ----
        model.eval()
        ae_mm = sse_mm = n_pix = 0.0
        hits = misses = fa = 0
        with torch.no_grad():
            for predictor, target_log, _target_mm in val_loader:
                predictor = predictor.to(device, non_blocking=True)
                target_log = target_log.to(device, non_blocking=True)
                bg_pred, ext_pred = model(predictor)
                combined_log = bg_pred + sign * args.beta * ext_pred

                # Convert both to mm for calibration metrics
                combined_mm = torch.exp(combined_log) - LOG_OFFSET
                target_mm = torch.exp(target_log) - LOG_OFFSET

                diff = (combined_mm - target_mm)
                ae_mm += diff.abs().sum().item()
                sse_mm += (diff ** 2).sum().item()
                n_pix += combined_mm.numel()

                pred_hot = combined_mm >= args.critical_threshold_mm
                gt_hot = target_mm >= args.critical_threshold_mm
                hits += (pred_hot & gt_hot).sum().item()
                misses += (~pred_hot & gt_hot).sum().item()
                fa += (pred_hot & ~gt_hot).sum().item()

        val_mae = ae_mm / max(n_pix, 1)
        val_rmse = math.sqrt(sse_mm / max(n_pix, 1))
        denom = hits + misses + fa
        val_csi = hits / denom if denom > 0 else 0.0
        print(
            f"Epoch {epoch+1} | train loss {sums['loss']/nb:.4f} "
            f"(bg={sums['bg']/nb:.4f} ext={sums['ext']/nb:.4f} cmb={sums['cmb']/nb:.4f}) "
            f"| val MAE {val_mae:.4f} mm  RMSE {val_rmse:.4f}  CSI@{args.critical_threshold_mm}mm {val_csi:.4f}"
        )

        improved = False
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_mae)
            improved = True
            print(f"   best-MAE saved -> {ckpt_mae.name}")
        if val_csi > best_csi:
            best_csi = val_csi
            torch.save(model.state_dict(), ckpt_csi)
            improved = True
            print(f"   best-CSI saved -> {ckpt_csi.name}")
        patience = 0 if improved else patience + 1
        if patience >= args.early_stopping_patience:
            print(f"   early stop after {patience} epochs without improvement")
            break

    print(f"\n-- done. best MAE={best_mae:.4f} mm  best CSI={best_csi:.4f} --")


if __name__ == "__main__":
    main()
