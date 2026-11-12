# Crowd: calibration map fitted after the patch sum

Script: `crowd/crowd_pool_then_calibrate.py`. Output: `crowd_postpool.json`. Computed on the released checkpoints (CSRNet and baseline, seeds 42–46).

## Control

The paper's crowd knob acts on pixels before the 32×32 patch sum, so its quantile-map and negative-shift members can reorder patches. Here the map is fitted on the validation split's pooled block values instead: the patch sums, or the further 2×2 or 4×4 pooling of the patch grid. It is the exact empirical quantile map from the forecast's pooled values onto the observed ones, with one map per arm and pooling, evaluated at the seven thresholds. Applied after pooling, it cannot reorder blocks, so it is exactly a per-arm threshold.

## Reproduction check

- The ground truth is identical: the positive-block counts match the released files in every cell.
- Raw CSI matches the released values to within 0.0018 in every cell. The residual differences are threshold flips of a few patches, caused by floating-point differences between this GPU and the original one.
- The released statistics are recomputed from the released files alone, and match the paper.

## Result

Spearman association between bias deficit and knob gain, with the paper's sign convention:

| Control | All 320 usable cells | Patch sums only (published protocol, n = 70) |
|---|---|---|
| Paper, per-cell knob applied before the patch sum | 0.568 | 0.497 |
| Paper, one global knob | 0.532 | 0.531 |
| Map fitted after pooling (no reordering possible) | **0.593** | **0.573** |

## Reading

The bias–gain relationship is slightly stronger when patch reordering is excluded by construction. The crowd result therefore does not depend on the knob changing the order of patches.
