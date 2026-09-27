"""
Training loop for ShanghaiTech SCDR experiment.

Supports:
- Baseline (ResNet + single decoder)
- Baseline + LDS (weighted sampling)
- Baseline + FDS (feature calibration on bottleneck features)
- Baseline + LDS + FDS
- CSRNet
- DualDecoder (dual decoder with frequency decomposition)

Methodology:
- Train on 270 images, validate on 30, evaluate on 182 test
- Select best epoch by VAL MAE (no test-set leakage)
- For DualDecoder, β is a deployment knob (full sweep at test, no β tuning)
- Multi-seed support: each call uses the provided seed for reproducibility
- Reports mean ± std across seeds in aggregation
"""
import os
# Required for CUDA deterministic ops with PyTorch >= 1.8
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from pathlib import Path
import numpy as np
import json
import random
import time

from dataset import ShanghaiTechDataset
from models.baseline import BaselineNet
from models.csrnet import CSRNet
from losses_modern import build_loss, NEEDS_POINTS, split_points_by_density
from dataset_points import ShanghaiTechPointsDataset, collate_with_points
from models.dual_decoder import DualDecoderNet, TieredWeightedMSELoss
from models.lds_fds import LDSMixin, FDSLayer
from evaluate import evaluate_model, format_results
import config


def set_seed(seed):
    """Set all relevant random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: torch.use_deterministic_algorithms(True) would be ideal but some
    # ops (e.g. ConvTranspose2d backward, scatter_add_) lack deterministic
    # implementations. Cross-seed variance dominates intra-seed variance,
    # so partial determinism is sufficient for valid mean ± std reporting.


def seed_worker(worker_id):
    """Seed each DataLoader worker deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """Create a torch generator for DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_scheduler(optimizer, total_epochs, warmup_epochs=None):
    """CosineAnnealing with linear warmup."""
    warmup_epochs = warmup_epochs if warmup_epochs is not None else config.WARMUP_EPOCHS
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def compute_count_bins(dataset, num_bins=100):
    """Compute integer bin index per sample based on integrated density (head count)."""
    counts = []
    for _, dm_path in dataset.samples:
        dm = np.load(dm_path)
        counts.append(dm.sum())
    counts = np.array(counts)
    bin_edges = np.linspace(counts.min() - 1, counts.max() + 1, num_bins + 1)
    bin_indices = np.digitize(counts, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    return bin_indices, bin_edges


class IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset to also yield each sample's index (for FDS bin lookup)."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, dm = self.base[idx]
        return img, dm, idx


def get_gradient_magnitudes(model, model_type):
    """Mean gradient magnitude for different model components."""
    grad_mags = {}
    if model_type == 'dual_decoder':
        bg_grads, ext_grads, enc_grads = [], [], []
        for name, param in model.named_parameters():
            if param.grad is not None:
                mag = param.grad.abs().mean().item()
                if 'bg_decoder' in name:
                    bg_grads.append(mag)
                elif 'ext_decoder' in name:
                    ext_grads.append(mag)
                elif 'encoder' in name:
                    enc_grads.append(mag)
        grad_mags['encoder'] = float(np.mean(enc_grads)) if enc_grads else 0.0
        grad_mags['bg_decoder'] = float(np.mean(bg_grads)) if bg_grads else 0.0
        grad_mags['ext_decoder'] = float(np.mean(ext_grads)) if ext_grads else 0.0
    else:
        decoder_grads, encoder_grads = [], []
        for name, param in model.named_parameters():
            if param.grad is not None:
                mag = param.grad.abs().mean().item()
                if 'encoder' in name:
                    encoder_grads.append(mag)
                else:
                    decoder_grads.append(mag)
        grad_mags['encoder'] = float(np.mean(encoder_grads)) if encoder_grads else 0.0
        grad_mags['decoder'] = float(np.mean(decoder_grads)) if decoder_grads else 0.0
    return grad_mags


def train_baseline(model_name='baseline', use_lds=False, use_fds=False, seed=42, tag_suffix='',
                   loss_name='mse', lr=None):
    """
    Train a single-decoder model with optional LDS/FDS.

    Methodology:
    - Train on 270 images (train split), validate on 30 held-out images (val split)
    - Select best epoch by VAL MAE (the standard crowd-counting metric)
    - Report final results on the 182-image TEST set using val-selected checkpoint
    - β is not relevant here (single decoder)
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = (model_name + ('_lds' if use_lds else '') + ('_fds' if use_fds else '')
           + ('' if loss_name == 'mse' else f'_{loss_name}') + tag_suffix)

    print(f"\n{'='*60}")
    print(f"Training: {tag} on {device}  [seed={seed}, val-based selection]")
    print(f"{'='*60}")

    # Datasets: train(270) / val(30) / test(182)
    # Bayesian Loss is defined over annotation points, so it needs the points
    # variant; every other arm sees byte-identical images from the base class.
    needs_points = loss_name in NEEDS_POINTS
    _TrainDS = ShanghaiTechPointsDataset if needs_points else ShanghaiTechDataset
    train_base = _TrainDS(
        config.DATA_ROOT, config.PART, 'train',
        crop_size=config.CROP_SIZE, augment=True
    )
    val_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'val',
        crop_size=config.CROP_SIZE, augment=False
    )
    test_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'test',
        crop_size=config.CROP_SIZE, augment=False
    )

    # FDS bin indices (computed on train split only)
    train_bin_indices = None
    if use_fds:
        train_bin_indices, _ = compute_count_bins(train_base, num_bins=100)
        train_bin_indices = torch.from_numpy(train_bin_indices).long()
        train_dataset = IndexedDataset(train_base)
    else:
        train_dataset = train_base

    # LDS sampler (weights computed from train split only)
    sampler = None
    if use_lds:
        lds_weights = LDSMixin.compute_lds_weights(train_base)
        sampler = WeightedRandomSampler(lds_weights, len(lds_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=(collate_with_points if needs_points else None)
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    if model_name == 'csrnet':
        model = CSRNet(pretrained=True).to(device)
    else:
        model = BaselineNet(pretrained=True, use_fds=use_fds).to(device)

    # DM-Count's count term is LINEAR in total predicted mass, so a randomly
    # initialised decoder head emitting ~13k persons produces an enormous first
    # gradient; at the shared LR that diverged on 2/5 seeds (epoch-1 train loss
    # 12,825 vs 387 on a healthy seed). Its authors specify lr=1e-5, so the arm
    # is additionally run at the published setting and both are reported.
    lr = config.LR if lr is None else lr
    criterion = build_loss(loss_name).to(device)
    # Balanced MSE carries a learnable noise scale; it must receive gradient or
    # the loss silently degenerates to a fixed-temperature softmax.
    loss_params = [p for p in criterion.parameters()] if isinstance(criterion, nn.Module) else []
    param_groups = [{'params': model.parameters()}]
    if loss_params:
        param_groups.append({'params': loss_params, 'lr': lr, 'weight_decay': 0.0})
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, config.EPOCHS)
    print(f"  loss = {loss_name}  lr = {lr:g}"
          + (f"  (+{len(loss_params)} learnable loss params)" if loss_params else ""))

    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    history = {'train_loss': [], 'val_eval': [], 'grad_mags': []}
    eval_every = 10
    best_val_mae = float('inf')
    best_epoch = None

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_grad_mags = []
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            pts = None
            if use_fds:
                imgs, gt_dms, idxs = batch
                bins = train_bin_indices[idxs].to(device)
            elif needs_points:
                imgs, gt_dms, pts = batch
                bins = None
            else:
                imgs, gt_dms = batch
                bins = None

            imgs = imgs.to(device)
            gt_dms = gt_dms.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_fds:
                pred = model(imgs, bin_indices=bins)
            else:
                pred = model(imgs)

            loss = criterion(pred, pts) if needs_points else criterion(pred, gt_dms)
            loss.backward()

            if batch_idx % 50 == 0:
                epoch_grad_mags.append(get_gradient_magnitudes(model, 'baseline'))

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_loss)
        if epoch_grad_mags:
            history['grad_mags'].append({k: float(np.mean([g[k] for g in epoch_grad_mags]))
                                         for k in epoch_grad_mags[0]})

        # Validate every N epochs
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if use_fds and hasattr(model, 'fds'):
                model.fds.calibrate()

            val_results = evaluate_model(
                model, val_loader, config.THRESHOLDS_A,
                patch_size=config.PATCH_SIZE, device=device
            )
            val_results['epoch'] = epoch + 1
            history['val_eval'].append(val_results)

            # Save checkpoint
            ckpt_path = ckpt_dir / f'epoch_{epoch+1:03d}.pth'
            torch.save(model.state_dict(), ckpt_path)

            # Track best by VAL MAE
            if val_results['mae'] < best_val_mae:
                best_val_mae = val_results['mae']
                best_epoch = epoch + 1

            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | TrainLoss: {avg_loss:.6f} | "
                  f"Val MAE: {val_results['mae']:7.2f} | "
                  f"Val CSI@5: {val_results['thresholds'][5.0]['csi']:.4f}"
                  + (" *" if val_results['mae'] == best_val_mae else ""))

    # Save history
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, default=str)

    print(f"\n  >> Best val epoch: {best_epoch} (val MAE={best_val_mae:.2f})")

    # Load val-selected checkpoint and evaluate on TEST set
    best_ckpt = ckpt_dir / f'epoch_{best_epoch:03d}.pth'
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    import shutil
    shutil.copy(best_ckpt, out_dir / 'best_model.pth')

    if use_fds and hasattr(model, 'fds'):
        model.fds.calibrate()

    test_results = evaluate_model(
        model, test_loader, config.THRESHOLDS_A,
        patch_size=config.PATCH_SIZE, device=device
    )
    test_results['epoch'] = best_epoch
    test_results['val_mae'] = best_val_mae
    test_results['seed'] = seed
    test_results['model_selection'] = 'val_mae'

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    print(format_results(test_results, tag + " (TEST)"))
    return test_results


def train_dual_decoder(tag='dual_decoder', decomp_sigma=None, alpha=None, beta_loss=None, delta=None,
               importance_mode='linear', use_lds=False, use_fds=False, epochs=None,
               seed=42, loss_name='mse', decomp_weight=1.0, tag_suffix='',
               decomp_mode='mse', split_tau=10.0):
    """
    Train DualDecoder dual-decoder (frequency decomposition).

    Methodology:
    - Train on 270 train images, validate on 30 val images.
    - Select best epoch by VAL MAE at β=1.0 (the training-time fusion).
    - At test time, sweep β as deployment knob and report full curve.
    - β at inference is NOT a tuned hyperparameter; it's a knob the user dials.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # A point-supervised loss (Bayesian Loss) needs annotation coordinates, so
    # the training set has to yield them; validation and test stay unchanged.
    tag = tag + tag_suffix
    needs_points = loss_name in NEEDS_POINTS
    criterion_pts = build_loss(loss_name).to(device) if loss_name != 'mse' else None
    lam = None

    decomp_sigma = decomp_sigma if decomp_sigma is not None else config.DECOMP_SIGMA
    alpha = alpha if alpha is not None else config.ALPHA
    beta_loss = beta_loss if beta_loss is not None else config.BETA_LOSS
    delta = delta if delta is not None else config.DELTA
    epochs = epochs if epochs is not None else config.EPOCHS

    print(f"\n{'='*60}")
    print(f"Training: {tag} on {device}  [seed={seed}, val-based selection]")
    print(f"  decomp_sigma={decomp_sigma}, α={alpha}, β_loss={beta_loss}, δ={delta}")
    print(f"  importance={importance_mode}, LDS={use_lds}, FDS={use_fds}, epochs={epochs}")
    print(f"{'='*60}")

    _TrainDS = ShanghaiTechPointsDataset if needs_points else ShanghaiTechDataset
    train_base = _TrainDS(
        config.DATA_ROOT, config.PART, 'train',
        crop_size=config.CROP_SIZE, augment=True
    )
    val_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'val',
        crop_size=config.CROP_SIZE, augment=False
    )
    test_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'test',
        crop_size=config.CROP_SIZE, augment=False
    )

    train_bin_indices = None
    if use_fds:
        train_bin_indices, _ = compute_count_bins(train_base, num_bins=100)
        train_bin_indices = torch.from_numpy(train_bin_indices).long()
        train_dataset = IndexedDataset(train_base)
    else:
        train_dataset = train_base

    sampler = None
    if use_lds:
        lds_weights = LDSMixin.compute_lds_weights(train_base)
        sampler = WeightedRandomSampler(lds_weights, len(lds_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=(collate_with_points if needs_points else None)
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = DualDecoderNet(pretrained=True, decomp_sigma=decomp_sigma, use_fds=use_fds).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, epochs)
    criterion_mse = nn.MSELoss(reduction='none')

    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    history = {'train_loss': [], 'loss_bg': [], 'loss_ext': [], 'loss_final': [],
               'val_eval': [], 'grad_mags': [],
               'config': {
                   'decomp_sigma': decomp_sigma, 'alpha': alpha,
                   'beta_loss': beta_loss, 'delta': delta,
                   'importance_mode': importance_mode,
                   'use_lds': use_lds, 'use_fds': use_fds,
                   'decomposition': 'frequency',
               }}
    eval_every = 10
    best_val_mae = float('inf')
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_loss_bg = epoch_loss_ext = epoch_loss_final = 0
        epoch_grad_mags = []
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            pts = None
            if use_fds:
                imgs, gt_dms, idxs = batch
                bins = train_bin_indices[idxs].to(device)
            elif needs_points:
                imgs, gt_dms, pts = batch
                bins = None
            else:
                imgs, gt_dms = batch
                bins = None

            imgs = imgs.to(device)
            gt_dms = gt_dms.to(device)

            # decompose_target is a per-sample scipy Gaussian on CPU and costs ~5x
            # the rest of the step, so skip it entirely when the decomposition terms
            # carry no weight (the pure architecture control). The targets are
            # unused there, so this changes nothing but wall-clock.
            need_decomp = (criterion_pts is None) or (
                decomp_mode == 'mse' and decomp_weight != 0.0)
            if need_decomp:
                with torch.no_grad():
                    gt_np = gt_dms.cpu().numpy()
                    bg_targets, ext_targets = [], []
                    for i in range(gt_np.shape[0]):
                        bg, ext = DualDecoderNet.decompose_target(gt_np[i, 0], sigma=decomp_sigma)
                        bg_targets.append(bg)
                        ext_targets.append(ext)
                    bg_target = torch.from_numpy(np.stack(bg_targets)).unsqueeze(1).to(device)
                    ext_target = torch.from_numpy(np.stack(ext_targets)).unsqueeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)
            final_pred, bg_pred, ext_pred = model(imgs, beta=1.0, bin_indices=bins)

            if need_decomp:
                loss_bg = criterion_mse(bg_pred, bg_target).mean()
                weights = DualDecoderNet.importance_weights(ext_target, mode=importance_mode)
                loss_ext = (criterion_mse(ext_pred, ext_target) * weights).mean()
                loss_final = criterion_mse(final_pred, gt_dms).mean()
            else:
                loss_bg = loss_ext = loss_final = final_pred.new_zeros(())

            mse_block = (alpha * loss_bg + beta_loss * loss_ext + delta * loss_final)
            if criterion_pts is None:
                combined_loss = mse_block
            else:
                # The fused head is supervised by the point loss while the
                # decomposition terms stay on MSE, and the two live on wildly
                # different scales (density-pixel MSE ~1e-6, Bayesian Loss ~1e2),
                # so any fixed weight would silently zero one of them out. lam is
                # calibrated ONCE on the first batch to put the MSE block at
                # `decomp_weight` times the point loss. decomp_weight=0 is then a
                # pure architecture control, with a byte-identical objective to the
                # single-decoder arm, only the decoder count differs.
                #
                # decomp_mode='points' avoids the mixing entirely: every head is
                # supervised by the SAME point loss, differing only in which
                # points it sees. A fixed lambda does not hold the two families
                # in balance anyway, since density-pixel MSE collapses far faster
                # than Bayesian Loss, so the decomposition block measured 46% of
                # the objective at epoch 1 and 3.9% by epoch 200, i.e. it decays
                # into a no-op and the arm silently becomes the architecture
                # control.
                loss_pts = criterion_pts(final_pred, pts)
                if decomp_mode == 'points':
                    bg_pts, ext_pts = split_points_by_density(
                        pts, gt_dms, tau=split_tau, patch=config.PATCH_SIZE)
                    loss_bg = criterion_pts(bg_pred, bg_pts)
                    loss_ext = criterion_pts(ext_pred, ext_pts)
                    combined_loss = (alpha * loss_bg + beta_loss * loss_ext
                                     + delta * loss_pts)
                elif decomp_weight == 0.0:
                    combined_loss = loss_pts
                else:
                    if lam is None:
                        lam = float(loss_pts.detach().abs()
                                    / mse_block.detach().abs().clamp_min(1e-12))
                        print(f"  point-loss / MSE-block scale lambda = {lam:.4g}", flush=True)
                    combined_loss = loss_pts + decomp_weight * lam * mse_block
            combined_loss.backward()

            if batch_idx % 50 == 0:
                epoch_grad_mags.append(get_gradient_magnitudes(model, 'dual_decoder'))

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += combined_loss.item()
            epoch_loss_bg += loss_bg.item()
            epoch_loss_ext += loss_ext.item()
            epoch_loss_final += loss_final.item()
            n_batches += 1

        scheduler.step()
        history['train_loss'].append(epoch_loss / n_batches)
        history['loss_bg'].append(epoch_loss_bg / n_batches)
        history['loss_ext'].append(epoch_loss_ext / n_batches)
        history['loss_final'].append(epoch_loss_final / n_batches)
        if epoch_grad_mags:
            history['grad_mags'].append({k: float(np.mean([g[k] for g in epoch_grad_mags]))
                                         for k in epoch_grad_mags[0]})

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if use_fds and hasattr(model, 'fds'):
                model.fds.calibrate()

            # Validate at β=1.0 only (training-time fusion)
            val_results = evaluate_model(
                model, val_loader, config.THRESHOLDS_A,
                patch_size=config.PATCH_SIZE, device=device,
                dual_decoder_beta=1.0
            )
            val_results['epoch'] = epoch + 1
            history['val_eval'].append(val_results)

            ckpt_path = ckpt_dir / f'epoch_{epoch+1:03d}.pth'
            torch.save(model.state_dict(), ckpt_path)

            if val_results['mae'] < best_val_mae:
                best_val_mae = val_results['mae']
                best_epoch = epoch + 1

            print(f"  Epoch {epoch+1:3d}/{epochs} | TrainLoss: {epoch_loss/n_batches:.6f} | "
                  f"Val(β=1.0) MAE: {val_results['mae']:7.2f} | "
                  f"Val CSI@5: {val_results['thresholds'][5.0]['csi']:.4f}"
                  + (" *" if val_results['mae'] == best_val_mae else ""))

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, default=str)

    print(f"\n  >> Best val epoch: {best_epoch} (val β=1.0 MAE={best_val_mae:.2f})")

    # Load val-selected checkpoint and run β sweep on TEST set
    best_ckpt = ckpt_dir / f'epoch_{best_epoch:03d}.pth'
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    import shutil
    shutil.copy(best_ckpt, out_dir / 'best_model.pth')

    if use_fds and hasattr(model, 'fds'):
        model.fds.calibrate()

    # Sweep β at test time as deployment knob
    test_betas = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    test_results = {}
    for beta in test_betas:
        r = evaluate_model(
            model, test_loader, config.THRESHOLDS_A,
            patch_size=config.PATCH_SIZE, device=device,
            dual_decoder_beta=beta
        )
        test_results[beta] = r
        print(format_results(r, tag, dual_decoder_beta=beta))

    test_results['best_epoch'] = best_epoch
    test_results['val_mae'] = best_val_mae
    test_results['seed'] = seed
    test_results['model_selection'] = 'val_mae_at_beta_1.0'

    with open(out_dir / 'results.json', 'w') as f:
        json.dump({str(k): v for k, v in test_results.items()}, f, indent=2, default=str)

    return test_results


def _density_sampler_weights(dataset, tau, base=0.1):
    """
    WeightedRandomSampler weights matching atmospheric DualDecoder.

    Each sample's weight = base + fraction of pixels with density >= tau.
    Samples with more critical pixels are oversampled.
    """
    weights = []
    for _, dm_path in dataset.samples:
        dm = np.load(dm_path)
        frac = float(np.mean(dm >= tau))
        weights.append(base + frac)
    return torch.tensor(weights, dtype=torch.float)


def train_dual_decoder_threshold(tag='dual_decoder_threshold', tau=0.005,
                          tier_thresholds=None,
                          alpha=0.5, beta_loss=1.5, delta=0.4,
                          use_sampler=True, epochs=None, seed=42):
    """
    Train DualDecoder with THRESHOLD-based decomposition matching the atmospheric setup.

    bg_target = min(y, tau), density capped at tau (smooth sub-critical field)
    ext_target = max(y - tau, 0), positive excess above tau (concentrated at peaks)

    Loss: alpha * MSE(cont, bg) + beta_loss * TieredMSE(ext, ext_target, y) + delta * MSE(cont+ext, y)
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = epochs if epochs is not None else config.EPOCHS

    if tier_thresholds is None:
        # Default: 10x weight for pixels >= 0.01, 25x for >= 0.05
        tier_thresholds = {0.01: 10.0, 0.05: 25.0}

    print(f"\n{'='*60}")
    print(f"Training: {tag} on {device}  [val-based epoch selection]")
    print(f"  THRESHOLD-based decomposition: tau={tau}")
    print(f"  Tiered loss thresholds: {tier_thresholds}")
    print(f"  Loss weights: alpha={alpha}, beta_loss={beta_loss}, delta={delta}")
    print(f"  Use density sampler: {use_sampler}")
    print(f"{'='*60}")

    train_base = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'train',
        crop_size=config.CROP_SIZE, augment=True
    )
    val_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'val',
        crop_size=config.CROP_SIZE, augment=False
    )
    test_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'test',
        crop_size=config.CROP_SIZE, augment=False
    )

    sampler = None
    if use_sampler:
        weights = _density_sampler_weights(train_base, tau=tau)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_base, batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = DualDecoderNet(pretrained=True, decomp_sigma=15.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, epochs)

    criterion_mse = nn.MSELoss()
    criterion_tiered = TieredWeightedMSELoss(thresholds=tier_thresholds)

    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    history = {'train_loss': [], 'loss_bg': [], 'loss_ext': [], 'loss_final': [],
               'val_eval': [], 'grad_mags': [],
               'config': {'tau': tau, 'tier_thresholds': tier_thresholds,
                          'alpha': alpha, 'beta_loss': beta_loss, 'delta': delta,
                          'use_sampler': use_sampler,
                          'decomposition': 'threshold'}}
    eval_every = 10
    best_val_mae = float('inf')
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_loss_bg = epoch_loss_ext = epoch_loss_final = 0
        epoch_grad_mags = []
        n_batches = 0

        for batch_idx, (imgs, gt_dms) in enumerate(train_loader):
            imgs = imgs.to(device)
            gt_dms = gt_dms.to(device)

            with torch.no_grad():
                bg_target = torch.clamp(gt_dms, max=tau)
                ext_target = gt_dms - bg_target

            optimizer.zero_grad(set_to_none=True)
            final_pred, bg_pred, ext_pred = model(imgs, beta=1.0)

            loss_bg = criterion_mse(bg_pred, bg_target)
            loss_ext = criterion_tiered(ext_pred, ext_target, gt_dms)
            loss_final = criterion_mse(final_pred, gt_dms)
            combined_loss = alpha * loss_bg + beta_loss * loss_ext + delta * loss_final

            combined_loss.backward()

            if batch_idx % 50 == 0:
                epoch_grad_mags.append(get_gradient_magnitudes(model, 'dual_decoder'))

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += combined_loss.item()
            epoch_loss_bg += loss_bg.item()
            epoch_loss_ext += loss_ext.item()
            epoch_loss_final += loss_final.item()
            n_batches += 1

        scheduler.step()
        history['train_loss'].append(epoch_loss / n_batches)
        history['loss_bg'].append(epoch_loss_bg / n_batches)
        history['loss_ext'].append(epoch_loss_ext / n_batches)
        history['loss_final'].append(epoch_loss_final / n_batches)
        if epoch_grad_mags:
            history['grad_mags'].append({k: float(np.mean([g[k] for g in epoch_grad_mags]))
                                         for k in epoch_grad_mags[0]})

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            val_results = evaluate_model(
                model, val_loader, config.THRESHOLDS_A,
                patch_size=config.PATCH_SIZE, device=device,
                dual_decoder_beta=1.0
            )
            val_results['epoch'] = epoch + 1
            history['val_eval'].append(val_results)

            ckpt_path = ckpt_dir / f'epoch_{epoch+1:03d}.pth'
            torch.save(model.state_dict(), ckpt_path)

            if val_results['mae'] < best_val_mae:
                best_val_mae = val_results['mae']
                best_epoch = epoch + 1

            print(f"  Epoch {epoch+1:3d}/{epochs} | TrainLoss: {epoch_loss/n_batches:.6f} | "
                  f"Val(β=1.0) MAE: {val_results['mae']:7.2f} | "
                  f"Val CSI@5: {val_results['thresholds'][5.0]['csi']:.4f}"
                  + (" *" if val_results['mae'] == best_val_mae else ""))

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, default=str)

    print(f"\n  >> Best val epoch: {best_epoch} (val β=1.0 MAE={best_val_mae:.2f})")

    best_ckpt = ckpt_dir / f'epoch_{best_epoch:03d}.pth'
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    import shutil
    shutil.copy(best_ckpt, out_dir / 'best_model.pth')

    test_betas = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    test_results = {}
    for beta in test_betas:
        r = evaluate_model(
            model, test_loader, config.THRESHOLDS_A,
            patch_size=config.PATCH_SIZE, device=device,
            dual_decoder_beta=beta
        )
        test_results[beta] = r
        print(format_results(r, tag, dual_decoder_beta=beta))

    test_results['best_epoch'] = best_epoch
    test_results['val_mae'] = best_val_mae
    test_results['seed'] = seed
    test_results['model_selection'] = 'val_mae_at_beta_1.0'

    with open(out_dir / 'results.json', 'w') as f:
        json.dump({str(k): v for k, v in test_results.items()}, f, indent=2, default=str)

    return test_results


def train_dual_decoder_unified(tag='dual_decoder_unified', tau=0.005,
                        alpha_ext=1.5, lambda_residual=10.0,
                        use_sampler=True, use_lds_on_ext=False,
                        epochs=None, seed=42):
    """
    Train DualDecoder with the UNIFIED decomposition:

      bg_target  = y                          (full field, no clipping)
      ext_target = clamp(y - tau, min=0)      (non-negative excess above tau)

    Loss:
      loss_bg  = MSE(bg_pred, y)
      loss_ext = mean((1 + lambda * ext_target) * (ext_pred - ext_target)^2)
      total    = loss_bg + alpha_ext * loss_ext

    Inference:
      ŷ = bg_pred + β * ext_pred

    Properties:
    - bg head trained as a single-decoder baseline (full field MSE)
    - ext head receives gradient ONLY at pixels where ext_target > 0 (critical region)
    - At β=0: ŷ = bg_pred ≈ baseline → competitive MAE
    - β > 0 progressively amplifies critical region predictions
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = epochs if epochs is not None else config.EPOCHS

    print(f"\n{'='*60}")
    print(f"Training: {tag} on {device}")
    print(f"  UNIFIED decomposition:")
    print(f"    bg_target  = y (full field)")
    print(f"    ext_target = clamp(y - {tau}, min=0)")
    print(f"  Loss: MSE(bg, y) + {alpha_ext} * weighted_MSE(ext, ext_target)")
    print(f"    weight = 1 + {lambda_residual} * ext_target")
    print(f"  Sampler: {'WeightedRandomSampler (density-based)' if use_sampler else 'uniform'}")
    print(f"{'='*60}")

    train_base = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'train',
        crop_size=config.CROP_SIZE, augment=True
    )
    val_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'val',
        crop_size=config.CROP_SIZE, augment=False
    )
    test_dataset = ShanghaiTechDataset(
        config.DATA_ROOT, config.PART, 'test',
        crop_size=config.CROP_SIZE, augment=False
    )

    sampler = None
    if use_sampler:
        weights = _density_sampler_weights(train_base, tau=tau)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_base, batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = DualDecoderNet(pretrained=True, decomp_sigma=15.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, epochs)
    criterion_mse = nn.MSELoss()
    criterion_mse_none = nn.MSELoss(reduction='none')

    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    history = {'train_loss': [], 'loss_bg': [], 'loss_ext': [],
               'val_eval': [], 'grad_mags': [],
               'config': {'tau': tau, 'alpha_ext': alpha_ext,
                          'lambda_residual': lambda_residual,
                          'use_sampler': use_sampler,
                          'decomposition': 'unified'}}
    eval_every = 10
    best_val_mae = float('inf')
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_loss_bg = epoch_loss_ext = 0
        epoch_grad_mags = []
        n_batches = 0

        for batch_idx, (imgs, gt_dms) in enumerate(train_loader):
            imgs = imgs.to(device)
            gt_dms = gt_dms.to(device)

            with torch.no_grad():
                bg_target = gt_dms
                ext_target = torch.clamp(gt_dms - tau, min=0)

            optimizer.zero_grad(set_to_none=True)
            final_pred, bg_pred, ext_pred = model(imgs, beta=1.0)

            loss_bg = criterion_mse(bg_pred, bg_target)
            with torch.no_grad():
                weights_t = 1.0 + lambda_residual * ext_target
            loss_ext = (criterion_mse_none(ext_pred, ext_target) * weights_t).mean()

            combined_loss = loss_bg + alpha_ext * loss_ext
            combined_loss.backward()

            if batch_idx % 50 == 0:
                epoch_grad_mags.append(get_gradient_magnitudes(model, 'dual_decoder'))

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += combined_loss.item()
            epoch_loss_bg += loss_bg.item()
            epoch_loss_ext += loss_ext.item()
            n_batches += 1

        scheduler.step()
        history['train_loss'].append(epoch_loss / n_batches)
        history['loss_bg'].append(epoch_loss_bg / n_batches)
        history['loss_ext'].append(epoch_loss_ext / n_batches)
        if epoch_grad_mags:
            history['grad_mags'].append({k: float(np.mean([g[k] for g in epoch_grad_mags]))
                                         for k in epoch_grad_mags[0]})

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            val_results = evaluate_model(
                model, val_loader, config.THRESHOLDS_A,
                patch_size=config.PATCH_SIZE, device=device,
                dual_decoder_beta=1.0
            )
            val_results['epoch'] = epoch + 1
            history['val_eval'].append(val_results)

            ckpt_path = ckpt_dir / f'epoch_{epoch+1:03d}.pth'
            torch.save(model.state_dict(), ckpt_path)

            if val_results['mae'] < best_val_mae:
                best_val_mae = val_results['mae']
                best_epoch = epoch + 1

            print(f"  Epoch {epoch+1:3d}/{epochs} | TrainLoss: {epoch_loss/n_batches:.6f} | "
                  f"Val(β=1.0) MAE: {val_results['mae']:7.2f} | "
                  f"Val CSI@5: {val_results['thresholds'][5.0]['csi']:.4f}"
                  + (" *" if val_results['mae'] == best_val_mae else ""))

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, default=str)

    print(f"\n  >> Best val epoch: {best_epoch} (val β=1.0 MAE={best_val_mae:.2f})")

    best_ckpt = ckpt_dir / f'epoch_{best_epoch:03d}.pth'
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    import shutil
    shutil.copy(best_ckpt, out_dir / 'best_model.pth')

    test_betas = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    test_results = {}
    for beta in test_betas:
        r = evaluate_model(
            model, test_loader, config.THRESHOLDS_A,
            patch_size=config.PATCH_SIZE, device=device,
            dual_decoder_beta=beta
        )
        test_results[beta] = r
        print(format_results(r, tag, dual_decoder_beta=beta))

    test_results['best_epoch'] = best_epoch
    test_results['val_mae'] = best_val_mae
    test_results['seed'] = seed
    test_results['model_selection'] = 'val_mae_at_beta_1.0'

    with open(out_dir / 'results.json', 'w') as f:
        json.dump({str(k): v for k, v in test_results.items()}, f, indent=2, default=str)

    return test_results


def run_all_experiments():
    all_results = {}
    all_results['baseline'] = train_baseline('baseline', use_lds=False, use_fds=False)
    all_results['baseline_lds'] = train_baseline('baseline', use_lds=True, use_fds=False)
    all_results['baseline_fds'] = train_baseline('baseline', use_lds=False, use_fds=True)
    all_results['baseline_lds_fds'] = train_baseline('baseline', use_lds=True, use_fds=True)
    all_results['csrnet'] = train_baseline('csrnet', use_lds=False, use_fds=False)
    all_results['dual_decoder'] = train_dual_decoder()

    out_path = config.OUTPUT_DIR / 'all_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")
    return all_results


if __name__ == '__main__':
    run_all_experiments()
