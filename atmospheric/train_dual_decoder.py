"""Train the DualDecoder with the unified decomposition.

Differences from train_unet.py:
  - Two losses (bg + α·ext), no "final" loss term
  - bg head trained on the FULL target with plain MSE
  - ext head trained on the non-negative critical residual with
    importance-weighted MSE (graded, no tier thresholds)
  - β is purely an inference-time knob. Training does NOT mix the two
    heads into a single combined prediction

Defaults are tuned for B08 atmosphere (low-value critical, τ=220K).

Example
-------
python atmospheric/train_dual_decoder.py \\
    --data-dir   $ATMOS_DATA/B08 \\
    --output-dir $RUNS/dual_decoder_B08_unified \\
    --band 8 --critical-threshold-k 220 --critical-mode low
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Dataset and model definitions are vendored under common/ so this runs from a
# clean checkout.
import sys as _s2
from pathlib import Path as _P2
_C = _P2(__file__).resolve().parents[1] / "common"
_s2.path[:0] = [str(_C), str(_C / "data"), str(_C / "models")]
from dataset import ALL_VARIABLES, MultiVariableARDataset, resolve_variable_indices
from losses.decomposition import (ImportanceWeightedMSELoss, SignedImportanceWeightedMSELoss,
                            decompose_frequency, decompose_unified)
from dual_decoder_unified import DualDecoderUnified


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # paths
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--band', type=int, default=8)

    # training hyperparameters
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--gradient-clip', type=float, default=1.0)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=None)

    # variable selection
    p.add_argument('--variables', type=str, default='T500,T850,RH700,W500',
                   help=f'Comma-separated subset of {ALL_VARIABLES}')

    # unified decomposition
    p.add_argument('--critical-threshold-k', type=float, default=220.0,
                   help='τ, the boundary of the critical region in Kelvin')
    p.add_argument('--decomposition', type=str, choices=['unified', 'frequency'],
                   default='unified',
                   help="Target decomposition. 'unified': bg=y, ext=relu(+-(tau-y)) -- a "
                        "pointwise function of y, so the ext head solves the same "
                        "prediction problem restricted to the tail. 'frequency': "
                        "bg=G_sigma*y, ext=y-G_sigma*y, neighbourhood-dependent, so the "
                        "ext head's task is distinct. Frequency is what the "
                        "crowd-density DualDecoder uses. Both yield a NON-NEGATIVE ext "
                        "target, as required by the ReLU ext head.")
    p.add_argument('--decomp-sigma', type=float, default=10.0,
                   help='Gaussian sigma (pixels) for --decomposition frequency.')
    p.add_argument('--critical-mode', type=str, choices=['low', 'high'], default='low',
                   help="'low'  = critical when y ≤ τ (atmosphere, cold cloud tops); "
                        "'high' = critical when y ≥ τ (e.g. crowd density)")

    # ---- β: now a TRAINING hyperparameter ----
    p.add_argument('--beta', type=float, default=1.0,
                   help='Training-time amplification of the ext head in the combined prediction. '
                        'The model is optimized so that ŷ = bg ± β·ext matches the full target. '
                        'Different β values produce different specialised models, with no inference sweep.')

    # loss weights: three terms now (bg + ext + combined)
    p.add_argument('--bg-weight', type=float, default=1.0,
                   help='Weight on the background-head MSE loss (bg head learns the full field)')
    p.add_argument('--ext-weight', type=float, default=1.0,
                   help='Weight on the importance-weighted ext-head loss (ext head learns the critical residual)')
    p.add_argument('--combined-weight', type=float, default=1.0,
                   help='Weight on the combined-prediction MSE loss at training β. '
                        'This is what couples the two heads. Set to 0 to recover the independent-heads training.')
    p.add_argument('--lambda-weight', type=float, default=10.0,
                   help='Importance-weight scale: w = 1 + λ · ext_target_norm')

    # sampler
    p.add_argument('--use-weighted-sampler', action='store_true', default=True)
    p.add_argument('--no-weighted-sampler', dest='use_weighted_sampler', action='store_false')
    p.add_argument('--sampling-weight-threshold', type=float, default=220.0)
    p.add_argument('--sampler-cache', type=Path, default=None)

    # Validation CSI tracking, the secondary metric for dual checkpointing
    p.add_argument('--val-csi-threshold-k', type=float, default=None,
                   help='Kelvin threshold for the val-set pixel-level CSI tracker. '
                        'Defaults to --critical-threshold-k if omitted.')

    p.add_argument('--device', type=str, default=None)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no-amp', dest='amp', action='store_false')

    return p.parse_args()


def get_sampler_weights(dataset, stats, threshold_k, cache_path):
    if cache_path.exists():
        print(f"Loading cached sampler weights from {cache_path}")
        return torch.load(cache_path)
    print(f"Pre-computing sampler weights (threshold {threshold_k}K)...")
    weights = []
    for _, target_norm, _ in tqdm(dataset, desc='sampler weights'):
        target_k = target_norm.numpy().squeeze() * (stats['target_std'] + 1e-8) + stats['target_mean']
        percentage = float(np.mean(target_k <= threshold_k))
        weights.append(0.1 + percentage)
    weights = torch.tensor(weights, dtype=torch.float)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, cache_path)
    return weights


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler_cache = args.sampler_cache or (args.output_dir / 'sampler_weights.pt')
    # Dual checkpoints: best by val MAE and best by val CSI.
    # The original filename is kept as a symlink/alias to the val-MAE checkpoint
    # so existing eval invocations continue to work without changes.
    model_save_path_mae = args.output_dir / f'dual_decoder_unified_B{args.band:02d}_best_mae.pth'
    model_save_path_csi = args.output_dir / f'dual_decoder_unified_B{args.band:02d}_best_csi.pth'
    model_save_path_alias = args.output_dir / f'dual_decoder_unified_B{args.band:02d}.pth'

    variable_names = [v.strip() for v in args.variables.split(',')]
    variable_indices = resolve_variable_indices(variable_names)
    input_channels = len(variable_indices)

    train_dataset = MultiVariableARDataset(args.data_dir, 'train')
    val_dataset = MultiVariableARDataset(args.data_dir, 'val')
    stats = train_dataset.stats
    std = stats['target_std'] + 1e-8
    mean = stats['target_mean']

    # Convert τ into normalized units once. The decomposition is computed
    # in normalized space because that's how targets and model outputs live.
    threshold_norm = (args.critical_threshold_k - mean) / std

    # Threshold for the val-CSI tracker (defaults to the decomposition τ)
    val_csi_threshold_k = (
        args.val_csi_threshold_k
        if args.val_csi_threshold_k is not None
        else args.critical_threshold_k
    )

    # Persist the resolved config so eval can reconstruct the threshold.
    config_path = args.output_dir / 'train_config.json'
    config_path.write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'variable_names': variable_names,
        'variable_indices': variable_indices,
        'input_channels': input_channels,
        'threshold_norm': float(threshold_norm),
        'target_mean': float(mean),
        'target_std': float(stats['target_std']),
        'device': str(device),
    }, indent=2))

    print(f"--- DualDecoder (unified) on band B{args.band:02d} | device={device} ---")
    print(f"Variables in: {variable_names} ({input_channels} channels)")
    print(f"τ = {args.critical_threshold_k}K  ({args.critical_mode}-value critical)")
    print(f"τ_norm = {threshold_norm:.4f}")
    print(f"β (training) = {args.beta}")
    print(f"Loss weights: bg={args.bg_weight}  ext={args.ext_weight}  combined={args.combined_weight}  λ={args.lambda_weight}")
    print(f"Weighted sampler: {args.use_weighted_sampler}")
    print(f"Val CSI tracker threshold: {val_csi_threshold_k}K")

    if args.use_weighted_sampler:
        sampler_weights = get_sampler_weights(
            train_dataset, stats, args.sampling_weight_threshold, sampler_cache,
        )
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = DualDecoderUnified(n_channels=input_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    bg_criterion = nn.MSELoss()
    # w = 1 + lambda*ext assumes ext >= 0; on a signed residual that yields
    # NEGATIVE weights, so the frequency path needs the signed variant.
    # Both decompositions now yield a NON-NEGATIVE critical residual, so the
    # standard w = 1 + lambda*ext weighting applies to each.
    ext_criterion = ImportanceWeightedMSELoss(lambda_weight=args.lambda_weight)
    combined_criterion = nn.MSELoss()

    # Sign of the ext head's contribution in the combined prediction:
    #   low-value critical (atmosphere)  →  ŷ = bg - β·ext
    #   high-value critical (crowds)     →  ŷ = bg + β·ext
    # For the unified decomposition ext_target is a non-negative magnitude, so the
    # critical direction has to be reinstated by the sign. The frequency residual
    # is already signed (cold anomalies are negative), so combining it with a
    # flipped sign would ANTI-sharpen. Hence sign = +1 there.
    sign = -1.0 if args.critical_mode == 'low' else 1.0

    best_val_mae = float('inf')
    best_val_csi = -1.0
    epochs_since_any_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_loss_bg = ep_loss_ext = ep_loss_combined = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train β={args.beta}]")
        for predictor, target_norm, _ in pbar:
            predictor_subset = predictor[:, variable_indices, :, :].to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                bg_pred, ext_pred = model(predictor_subset)

                # Decomposition lives entirely in normalized space. The frequency
                # residual is scale-equivariant under the affine normalization
                # (the mean cancels), so this matches decomposing in Kelvin.
                if args.decomposition == 'frequency':
                    _, ext_target_norm = decompose_frequency(
                        target_norm, args.decomp_sigma, mode=args.critical_mode)
                else:
                    _, ext_target_norm = decompose_unified(
                        target_norm, threshold_norm, mode=args.critical_mode,
                    )

                # Combined prediction at the training β (the actual operating
                # point this model is being optimized for)
                combined_pred = bg_pred + sign * args.beta * ext_pred

                loss_bg = bg_criterion(bg_pred, target_norm)
                loss_ext = ext_criterion(ext_pred, ext_target_norm)
                loss_combined = combined_criterion(combined_pred, target_norm)

                loss = (
                    args.bg_weight * loss_bg
                    + args.ext_weight * loss_ext
                    + args.combined_weight * loss_combined
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.item())
            ep_loss_bg += float(loss_bg.item())
            ep_loss_ext += float(loss_ext.item())
            ep_loss_combined += float(loss_combined.item())
            n_batches += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bg': f"{loss_bg.item():.4f}",
                'ext': f"{loss_ext.item():.4f}",
                'cmb': f"{loss_combined.item():.4f}",
            })

        # ----- validation: track combined-prediction MAE AND pixel-wise CSI ----
        model.eval()
        sse = ae = n_pix = 0
        # Pixel-level contingency tally for val CSI
        hits = misses = false_alarms = 0
        with torch.no_grad():
            for predictor, target_norm, _ in val_loader:
                predictor_subset = predictor[:, variable_indices, :, :].to(device, non_blocking=True)
                target_norm = target_norm.to(device, non_blocking=True)
                bg_pred, ext_pred = model(predictor_subset)
                combined_norm = bg_pred + sign * args.beta * ext_pred
                combined_k = combined_norm * std + mean
                target_k = target_norm * std + mean

                # MAE / RMSE
                ae += float((combined_k - target_k).abs().sum().item())
                sse += float(((combined_k - target_k) ** 2).sum().item())
                n_pix += int(combined_k.numel())

                # CSI at val_csi_threshold_k (pixel-level, low-value critical)
                pred_cold = combined_k <= val_csi_threshold_k
                gt_cold = target_k <= val_csi_threshold_k
                hits += int((pred_cold & gt_cold).sum().item())
                misses += int((~pred_cold & gt_cold).sum().item())
                false_alarms += int((pred_cold & ~gt_cold).sum().item())

        val_mae = ae / max(n_pix, 1)
        val_rmse = (sse / max(n_pix, 1)) ** 0.5
        denom = hits + misses + false_alarms
        val_csi = hits / denom if denom > 0 else 0.0
        avg_train_loss = ep_loss / max(n_batches, 1)

        print(
            f"Epoch {epoch + 1} | train loss: {avg_train_loss:.6f} "
            f"(bg={ep_loss_bg / n_batches:.4f}  ext={ep_loss_ext / n_batches:.4f}  "
            f"cmb={ep_loss_combined / n_batches:.4f})  "
            f"| val MAE: {val_mae:.4f}K  CSI@{val_csi_threshold_k:g}K: {val_csi:.4f}"
        )

        improved_any = False
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path_mae)
            # Keep the legacy filename pointed at the best-MAE checkpoint so
            # eval scripts that hard-code it still work.
            torch.save(model.state_dict(), model_save_path_alias)
            print(f"   ✅ best-MAE model saved to {model_save_path_mae.name}")
            improved_any = True

        if val_csi > best_val_csi:
            best_val_csi = val_csi
            torch.save(model.state_dict(), model_save_path_csi)
            print(f"   ✅ best-CSI model saved to {model_save_path_csi.name}")
            improved_any = True

        if improved_any:
            epochs_since_any_improvement = 0
        else:
            epochs_since_any_improvement += 1
            if epochs_since_any_improvement >= args.early_stopping_patience:
                print(
                    f"   early stopping after {epochs_since_any_improvement} epochs "
                    f"since either val MAE or val CSI improved"
                )
                break

    print(
        f"\n--- Training finished. "
        f"Best val MAE: {best_val_mae:.4f}K  |  Best val CSI@{val_csi_threshold_k:g}K: {best_val_csi:.4f} ---"
    )


if __name__ == '__main__':
    main()
