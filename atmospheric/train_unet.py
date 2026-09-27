"""Train the Attention U-Net baseline on a per-band AR-downscaling dataset.

Reproduces UNet_Retraining.ipynb verbatim, with all hyperparameters and
paths exposed as CLI arguments. Defaults match the published B08 config.

Example
-------
python train.py \\
    --data-dir $ATMOS_DATA/B08 \\
    --output-dir $RUNS/unet_B08 \\
    --band 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
from losses import TieredWeightedMSELoss
from attention_unet import AttentionUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--band', type=int, default=8)

    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--gradient-clip', type=float, default=1.0)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=None)

    p.add_argument('--variables', type=str, default='T500,T850,RH700,W500',
                   help=f'Comma-separated subset of {ALL_VARIABLES} (default omits IVT)')

    # model hyperparameters
    p.add_argument('--base-channels', type=int, default=64)
    p.add_argument('--unet-depth', type=int, default=4)
    p.add_argument('--no-attention', action='store_true', help='Disable bottleneck attention')

    # loss
    p.add_argument('--loss-thresholds', type=str, default='220:10,210:25')

    # sampler
    p.add_argument('--sampling-weight-threshold', type=float, default=220.0)
    p.add_argument('--sampler-cache', type=Path, default=None)

    p.add_argument('--device', type=str, default=None)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no-amp', dest='amp', action='store_false')

    return p.parse_args()


def parse_loss_thresholds(s):
    out = {}
    for chunk in s.split(','):
        k, v = chunk.split(':')
        out[float(k)] = float(v)
    return out


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
    model_save_path = args.output_dir / f'attention_unet_B{args.band:02d}.pth'

    variable_names = [v.strip() for v in args.variables.split(',')]
    variable_indices = resolve_variable_indices(variable_names)
    input_channels = len(variable_indices)
    loss_thresholds = parse_loss_thresholds(args.loss_thresholds)

    config_path = args.output_dir / 'train_config.json'
    config_path.write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'variable_names': variable_names,
        'variable_indices': variable_indices,
        'input_channels': input_channels,
        'loss_thresholds': loss_thresholds,
        'device': str(device),
    }, indent=2))

    print(f"--- Attention U-Net training on band B{args.band:02d} | device={device} ---")
    print(f"Variables in: {variable_names} ({input_channels} channels)")
    print(f"Tiered loss thresholds: {loss_thresholds}")

    train_dataset = MultiVariableARDataset(args.data_dir, 'train')
    val_dataset = MultiVariableARDataset(args.data_dir, 'val')

    sampler_weights = get_sampler_weights(
        train_dataset, train_dataset.stats, args.sampling_weight_threshold, sampler_cache,
    )
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = AttentionUNet(
        input_channels=input_channels,
        base_channels=args.base_channels,
        depth=args.unet_depth,
        use_attention=not args.no_attention,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = TieredWeightedMSELoss(thresholds=loss_thresholds)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for predictor, target_norm, _ in pbar:
            predictor_subset = predictor[:, variable_indices, :, :].to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction_norm = model(predictor_subset)
                loss = criterion(prediction_norm, target_norm, train_dataset.stats)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for predictor, target_norm, _ in val_loader:
                predictor_subset = predictor[:, variable_indices, :, :].to(device, non_blocking=True)
                target_norm = target_norm.to(device, non_blocking=True)
                prediction_norm = model(predictor_subset)
                val_total += criterion(prediction_norm, target_norm, val_dataset.stats).item()

        avg_val_loss = val_total / len(val_loader)
        print(f"Epoch {epoch + 1} | val loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   ✅ best model saved to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"   🛑 early stopping after {patience_counter} epochs without improvement")
                break

    print(f"\n--- Training finished. Best val loss: {best_val_loss:.6f} ---")


if __name__ == '__main__':
    main()
