"""
Evaluation metrics for ShanghaiTech SCDR experiment.

Metrics:
- MAE: Mean Absolute Error on integrated counts
- CSI: Critical Success Index at density threshold (patch-level)
- Frequency Bias: predicted/true exceedance ratio
- Stratified MAE: MAE split by high-density vs background patches
"""
import numpy as np
import torch


def compute_mae(pred_density, gt_density):
    """
    MAE on integrated counts (standard crowd counting metric).

    Args:
        pred_density: (H, W) predicted density map
        gt_density: (H, W) ground truth density map

    Returns:
        mae: absolute error in total count
    """
    return abs(pred_density.sum() - gt_density.sum())


def compute_patch_metrics(pred_density, gt_density, threshold, patch_size=32):
    """
    Compute patch-level CSI, Bias, and stratified MAE.

    Divides the image into non-overlapping patches, integrates density
    per patch, applies threshold to determine exceedance.

    Args:
        pred_density: (H, W) predicted density map
        gt_density: (H, W) ground truth density map
        threshold: persons per patch threshold for "critical"
        patch_size: size of evaluation patches

    Returns:
        dict with keys: hits, misses, false_alarms, csi, bias,
                        mae_high, mae_low, n_high, n_low, R
    """
    h, w = gt_density.shape

    pred_patches = []
    gt_patches = []

    for yi in range(0, h - patch_size + 1, patch_size):
        for xi in range(0, w - patch_size + 1, patch_size):
            pred_patches.append(pred_density[yi:yi+patch_size, xi:xi+patch_size].sum())
            gt_patches.append(gt_density[yi:yi+patch_size, xi:xi+patch_size].sum())

    pred_patches = np.array(pred_patches)
    gt_patches = np.array(gt_patches)

    # Binary exceedance
    pred_exceed = pred_patches > threshold
    gt_exceed = gt_patches > threshold

    hits = int(np.sum(pred_exceed & gt_exceed))
    misses = int(np.sum(~pred_exceed & gt_exceed))
    false_alarms = int(np.sum(pred_exceed & ~gt_exceed))

    # CSI
    denom = hits + misses + false_alarms
    csi = hits / denom if denom > 0 else float('nan')

    # Frequency Bias
    bias_denom = hits + misses
    bias = (hits + false_alarms) / bias_denom if bias_denom > 0 else float('nan')

    # R: fraction of patches exceeding threshold in ground truth
    n_total = len(gt_patches)
    n_gt_exceed = int(np.sum(gt_exceed))
    R = n_gt_exceed / n_total if n_total > 0 else 0.0

    # Stratified MAE (patch-level count error)
    high_mask = gt_exceed
    low_mask = ~gt_exceed

    mae_high = float(np.mean(np.abs(pred_patches[high_mask] - gt_patches[high_mask]))) if high_mask.any() else float('nan')
    mae_low = float(np.mean(np.abs(pred_patches[low_mask] - gt_patches[low_mask]))) if low_mask.any() else float('nan')

    return {
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'csi': csi,
        'bias': bias,
        'R': R,
        'n_high': int(np.sum(high_mask)),
        'n_low': int(np.sum(low_mask)),
        'mae_high': mae_high,
        'mae_low': mae_low,
    }


def evaluate_model(model, dataloader, thresholds, patch_size=32,
                   device='cuda', dual_decoder_beta=None):
    """
    Full evaluation of a model on a test set.

    Args:
        model: trained model (BaselineNet, CSRNet, or DualDecoderNet)
        dataloader: test DataLoader (batch_size=1 for variable image sizes)
        thresholds: list of threshold values for CSI evaluation
        patch_size: patch size for CSI
        device: torch device
        dual_decoder_beta: if not None, model is DualDecoder and this beta is used

    Returns:
        results: dict with aggregate and per-threshold metrics
    """
    model.eval()
    is_dual_decoder = dual_decoder_beta is not None

    all_mae = []
    per_threshold = {t: {'hits': 0, 'misses': 0, 'false_alarms': 0,
                         'mae_high_list': [], 'mae_low_list': [],
                         'R_list': [], 'n_high_total': 0, 'n_low_total': 0}
                     for t in thresholds}

    with torch.no_grad():
        for img, gt_dm in dataloader:
            img = img.to(device)

            if is_dual_decoder:
                final_pred, _, _ = model(img, beta=dual_decoder_beta)
                pred = final_pred
            else:
                pred = model(img)

            # Convert to numpy (remove batch and channel dims)
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt_dm.squeeze().cpu().numpy()

            # Ensure non-negative predictions
            pred_np = np.maximum(pred_np, 0)

            # MAE (count-level)
            mae = compute_mae(pred_np, gt_np)
            all_mae.append(mae)

            # Patch-level metrics at each threshold
            for t in thresholds:
                pm = compute_patch_metrics(pred_np, gt_np, t, patch_size)
                per_threshold[t]['hits'] += pm['hits']
                per_threshold[t]['misses'] += pm['misses']
                per_threshold[t]['false_alarms'] += pm['false_alarms']
                per_threshold[t]['R_list'].append(pm['R'])
                per_threshold[t]['n_high_total'] += pm['n_high']
                per_threshold[t]['n_low_total'] += pm['n_low']
                if not np.isnan(pm['mae_high']):
                    per_threshold[t]['mae_high_list'].append(pm['mae_high'])
                if not np.isnan(pm['mae_low']):
                    per_threshold[t]['mae_low_list'].append(pm['mae_low'])

    # Aggregate results
    results = {
        'mae': float(np.mean(all_mae)),
        'mae_std': float(np.std(all_mae)),
        'thresholds': {}
    }

    for t in thresholds:
        pt = per_threshold[t]
        h, m, fa = pt['hits'], pt['misses'], pt['false_alarms']
        denom = h + m + fa
        csi = h / denom if denom > 0 else 0.0
        bias_denom = h + m
        bias = (h + fa) / bias_denom if bias_denom > 0 else float('nan')

        results['thresholds'][t] = {
            'csi': csi,
            'bias': bias,
            'hits': h,
            'misses': m,
            'false_alarms': fa,
            'mean_R': float(np.mean(pt['R_list'])),
            'mae_high': float(np.mean(pt['mae_high_list'])) if pt['mae_high_list'] else float('nan'),
            'mae_low': float(np.mean(pt['mae_low_list'])) if pt['mae_low_list'] else float('nan'),
        }

    return results


def evaluate_model_fds_two_pass(model, dataloader, thresholds, bin_edges,
                                num_bins=100, patch_size=32, device='cuda'):
    """
    Two-pass evaluation for FDS-equipped models.

    Pass 1: predict counts with bin_indices=None (uncalibrated path).
    Pass 2: bin pass-1 counts using train-set bin_edges, re-predict with
    calibration applied. Pass-2 prediction is what we score.

    Args:
        model: BaselineNet with use_fds=True and a calibrated FDSLayer
        dataloader: test DataLoader (batch_size=1)
        thresholds: CSI thresholds
        bin_edges: numpy array (num_bins+1,) from compute_count_bins on train split
        num_bins: number of bins (must match training-time num_bins)
    """
    model.eval()
    all_mae = []
    per_threshold = {t: {'hits': 0, 'misses': 0, 'false_alarms': 0,
                         'mae_high_list': [], 'mae_low_list': [],
                         'R_list': [], 'n_high_total': 0, 'n_low_total': 0}
                     for t in thresholds}

    with torch.no_grad():
        for img, gt_dm in dataloader:
            img = img.to(device)

            # Pass 1: uncalibrated → count estimate
            pred1 = model(img, bin_indices=None)
            count_est = float(pred1.sum().clamp(min=0).item())

            # Bin the predicted count using training-set edges
            bin_idx = int(np.digitize([count_est], bin_edges)[0] - 1)
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            bin_indices = torch.tensor([bin_idx], dtype=torch.long, device=device)

            # Pass 2: with calibration
            pred = model(img, bin_indices=bin_indices)

            pred_np = np.maximum(pred.squeeze().cpu().numpy(), 0)
            gt_np = gt_dm.squeeze().cpu().numpy()

            all_mae.append(compute_mae(pred_np, gt_np))
            for t in thresholds:
                pm = compute_patch_metrics(pred_np, gt_np, t, patch_size)
                per_threshold[t]['hits'] += pm['hits']
                per_threshold[t]['misses'] += pm['misses']
                per_threshold[t]['false_alarms'] += pm['false_alarms']
                per_threshold[t]['R_list'].append(pm['R'])
                per_threshold[t]['n_high_total'] += pm['n_high']
                per_threshold[t]['n_low_total'] += pm['n_low']
                if not np.isnan(pm['mae_high']):
                    per_threshold[t]['mae_high_list'].append(pm['mae_high'])
                if not np.isnan(pm['mae_low']):
                    per_threshold[t]['mae_low_list'].append(pm['mae_low'])

    results = {
        'mae': float(np.mean(all_mae)),
        'mae_std': float(np.std(all_mae)),
        'thresholds': {},
    }
    for t in thresholds:
        pt = per_threshold[t]
        h, m, fa = pt['hits'], pt['misses'], pt['false_alarms']
        denom = h + m + fa
        csi = h / denom if denom > 0 else 0.0
        bias_denom = h + m
        bias = (h + fa) / bias_denom if bias_denom > 0 else float('nan')
        results['thresholds'][t] = {
            'csi': csi, 'bias': bias, 'hits': h, 'misses': m, 'false_alarms': fa,
            'mean_R': float(np.mean(pt['R_list'])),
            'mae_high': float(np.mean(pt['mae_high_list'])) if pt['mae_high_list'] else float('nan'),
            'mae_low': float(np.mean(pt['mae_low_list'])) if pt['mae_low_list'] else float('nan'),
        }
    return results


def format_results(results, model_name, dual_decoder_beta=None):
    """Pretty-print evaluation results."""
    header = f"{model_name}"
    if dual_decoder_beta is not None:
        header += f" (beta={dual_decoder_beta})"

    lines = [
        f"\n{'='*60}",
        header,
        f"{'='*60}",
        f"  MAE: {results['mae']:.2f} +/- {results['mae_std']:.2f}",
    ]

    for t, tr in results['thresholds'].items():
        lines.append(f"\n  Threshold = {t:.1f} persons/patch (mean R = {tr['mean_R']:.4f}):")
        lines.append(f"    CSI:  {tr['csi']:.4f}  |  Bias: {tr['bias']:.4f}")
        lines.append(f"    Hits: {tr['hits']}  Misses: {tr['misses']}  FA: {tr['false_alarms']}")
        lines.append(f"    MAE(high): {tr['mae_high']:.4f}  MAE(low): {tr['mae_low']:.4f}")

    return '\n'.join(lines)
