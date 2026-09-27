"""Reference implementation of the FreeKnob Audit protocol.

The monotone quantile map is a classical calibration instrument. The protocol
implemented here is the paper's contribution: fit one map for every arm on the
same disjoint calibration split, apply those fixed maps on the evaluation split,
and report pooled frequency bias together with every raw and calibrated contrast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class QuantileMap:
    """A monotone piecewise-linear map between two empirical marginals."""

    source_quantiles: np.ndarray
    target_quantiles: np.ndarray

    def __call__(self, values: ArrayLike) -> np.ndarray:
        """Apply the map, linearly extrapolating beyond the fitted range."""
        source = self.source_quantiles
        target = self.target_quantiles
        x = np.asarray(values, dtype=source.dtype)
        if source.size == 1:
            return np.full_like(x, target[0], dtype=target.dtype)

        flat = x.reshape(-1)
        index = np.searchsorted(source, flat, side="left")
        index = np.clip(index, 1, source.size - 1)
        x0, x1 = source[index - 1], source[index]
        y0, y1 = target[index - 1], target[index]
        weight = np.divide(
            flat - x0,
            x1 - x0,
            out=np.zeros_like(flat),
            where=x1 > x0,
        )
        return (y0 + weight * (y1 - y0)).reshape(x.shape)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "source_quantiles": self.source_quantiles.tolist(),
            "target_quantiles": self.target_quantiles.tolist(),
        }


@dataclass(frozen=True)
class ContingencyScore:
    hits: int
    false_alarms: int
    misses: int
    csi: float
    frequency_bias: float
    deviation: Optional[float]


@dataclass(frozen=True)
class ArmAudit:
    raw: ContingencyScore
    calibrated: ContingencyScore
    csi_gain: float


@dataclass(frozen=True)
class PairAudit:
    arm_a: str
    arm_b: str
    raw_contrast: float
    calibrated_contrast: float
    distortion: float
    calibration_gap: Optional[float]
    ranking_reversal: bool


@dataclass
class AuditReport:
    """Results and fitted maps from one matched audit."""

    threshold: float
    pooling: str
    block_size: int
    arms: dict[str, ArmAudit]
    pairs: list[PairAudit]
    calibrators: dict[str, QuantileMap] = field(repr=False)

    def to_dict(self, include_calibrators: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "protocol": "FreeKnob Audit",
            "threshold": self.threshold,
            "pooling": self.pooling,
            "block_size": self.block_size,
            "arms": {name: asdict(audit) for name, audit in self.arms.items()},
            "pairs": [asdict(pair) for pair in self.pairs],
        }
        if include_calibrators:
            result["calibrators"] = {
                name: calibrator.to_dict()
                for name, calibrator in self.calibrators.items()
            }
        return result

    def to_json(self, path: Optional[Union[str, Path]] = None, *,
                include_calibrators: bool = False, indent: int = 2) -> str:
        text = json.dumps(
            self.to_dict(include_calibrators=include_calibrators), indent=indent
        )
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


def _finite_sample(values: ArrayLike, max_samples: Optional[int],
                   rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("calibration arrays must contain at least one finite value")
    if max_samples is not None and flat.size > max_samples:
        flat = flat[rng.choice(flat.size, size=max_samples, replace=False)]
    return flat


def fit_quantile_map(predictions: ArrayLike, observations: ArrayLike, *,
                     n_quantiles: int = 1001,
                     max_samples: Optional[int] = 60_000_000,
                     seed: int = 0) -> QuantileMap:
    """Fit one global marginal quantile map on calibration data only."""
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be at least 2 or None")
    rng = np.random.default_rng(seed)
    pred = _finite_sample(predictions, max_samples, rng)
    obs = _finite_sample(observations, max_samples, rng)
    probabilities = np.linspace(0.0, 100.0, n_quantiles)
    source = np.maximum.accumulate(np.percentile(pred, probabilities))
    target = np.maximum.accumulate(np.percentile(obs, probabilities))
    # The paper's reference implementation stores and applies its knots in
    # float32.  Keeping that contract also prevents average pooling from moving
    # a handful of values across integer VIL thresholds on different backends.
    return QuantileMap(source.astype(np.float32), target.astype(np.float32))


def spatial_pool(values: ArrayLike, pooling: str = "none",
                 block_size: int = 1) -> np.ndarray:
    """Reduce non-overlapping blocks over the final two dimensions.

    Remainders are dropped, matching floor-stride pooling in the paper's
    PyTorch implementation. ``avg`` and ``average`` are aliases for ``mean``.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    aliases = {"avg": "mean", "average": "mean", "identity": "none"}
    pooling = aliases.get(pooling, pooling)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if pooling == "none":
        return array
    if pooling not in {"mean", "max"}:
        raise ValueError("pooling must be one of: none, mean, max")
    if array.ndim < 2:
        raise ValueError("spatial pooling requires arrays with at least two dimensions")

    height, width = array.shape[-2:]
    pooled_height, pooled_width = height // block_size, width // block_size
    if pooled_height == 0 or pooled_width == 0:
        raise ValueError("block_size is larger than an input spatial dimension")
    cropped = array[..., :pooled_height * block_size,
                    :pooled_width * block_size]
    blocks = cropped.reshape(
        *cropped.shape[:-2], pooled_height, block_size, pooled_width, block_size
    )
    if pooling == "max":
        return blocks.max(axis=(-3, -1))

    # Match torch.nn.functional.avg_pool2d: accumulate each block in row-major
    # order at the input floating-point precision, then divide.  A multi-axis
    # NumPy mean may choose another reduction tree; on SEVIR that was enough to
    # move two of ~12 million pooled values across a threshold.
    pooled = np.zeros(
        (*cropped.shape[:-2], pooled_height, pooled_width), dtype=array.dtype
    )
    for row in range(block_size):
        for column in range(block_size):
            pooled += blocks[..., :, row, :, column]
    pooled /= block_size * block_size
    return pooled


def contingency_score(predictions: ArrayLike, observations: ArrayLike,
                      threshold: float) -> ContingencyScore:
    """Score already pooled arrays at one threshold."""
    pred = np.asarray(predictions)
    obs = np.asarray(observations)
    if pred.shape != obs.shape:
        raise ValueError(
            f"prediction and observation shapes differ: "
            f"{pred.shape} != {obs.shape}"
        )
    if pred.size == 0:
        raise ValueError("prediction and observation arrays must not be empty")
    if not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError("evaluation arrays must contain only finite values")
    pred_positive = pred >= threshold
    obs_positive = obs >= threshold
    hits = int(np.count_nonzero(pred_positive & obs_positive))
    false_alarms = int(np.count_nonzero(pred_positive & ~obs_positive))
    misses = int(np.count_nonzero(~pred_positive & obs_positive))
    csi_denominator = hits + false_alarms + misses
    event_denominator = hits + misses
    csi = hits / max(csi_denominator, 1)
    bias = (hits + false_alarms) / max(event_denominator, 1)
    deviation = abs(math.log(bias)) if bias > 0 else None
    return ContingencyScore(
        hits, false_alarms, misses, csi, bias, deviation
    )


def threshold_pool_score(predictions: ArrayLike, observations: ArrayLike,
                         threshold: float, *, pooling: str = "none",
                         block_size: int = 1) -> ContingencyScore:
    """Pool continuous fields, threshold them, and score their contingency table."""
    pred = spatial_pool(predictions, pooling, block_size)
    obs = spatial_pool(observations, pooling, block_size)
    return contingency_score(pred, obs, threshold)


def free_knob_audit(
    predictions_calibration: Mapping[str, ArrayLike],
    observations_calibration: ArrayLike,
    predictions_test: Mapping[str, ArrayLike],
    observations_test: ArrayLike,
    *,
    threshold: float,
    pooling: str = "none",
    block_size: int = 1,
    n_quantiles: int = 1001,
    max_fit_samples: Optional[int] = 60_000_000,
    seed: int = 0,
) -> AuditReport:
    """Run the matched-arm, disjoint-split FreeKnob Audit.

    The caller owns the split: ``*_calibration`` is used only to fit each arm's
    map, and all reported scores use ``*_test``. Every arm must appear in both
    prediction mappings. Pair signs follow the paper: for pair (A, B), contrast
    is CSI(B)-CSI(A), distortion is raw-calibrated, and calibration gap is
    dev(A)-dev(B), where dev is absolute log frequency bias.
    """
    calibration_names = list(predictions_calibration)
    test_names = list(predictions_test)
    if not calibration_names:
        raise ValueError("at least one arm is required")
    if set(calibration_names) != set(test_names):
        raise ValueError("calibration and test mappings must contain the same arms")

    obs_cal = np.asarray(observations_calibration)
    obs_test = np.asarray(observations_test)
    calibrators: dict[str, QuantileMap] = {}
    arms: dict[str, ArmAudit] = {}
    for name in calibration_names:
        pred_cal = np.asarray(predictions_calibration[name])
        pred_test = np.asarray(predictions_test[name])
        if pred_cal.shape != obs_cal.shape:
            raise ValueError(
                f"calibration shape for {name!r} is {pred_cal.shape}; "
                f"expected {obs_cal.shape}"
            )
        if pred_test.shape != obs_test.shape:
            raise ValueError(
                f"test shape for {name!r} is {pred_test.shape}; "
                f"expected {obs_test.shape}"
            )
        calibrator = fit_quantile_map(
            pred_cal,
            obs_cal,
            n_quantiles=n_quantiles,
            max_samples=max_fit_samples,
            seed=seed,
        )
        raw = threshold_pool_score(
            pred_test, obs_test, threshold, pooling=pooling,
            block_size=block_size
        )
        calibrated = threshold_pool_score(
            calibrator(pred_test), obs_test, threshold, pooling=pooling,
            block_size=block_size
        )
        calibrators[name] = calibrator
        arms[name] = ArmAudit(raw, calibrated, calibrated.csi - raw.csi)

    pairs = []
    for arm_a, arm_b in combinations(calibration_names, 2):
        score_a, score_b = arms[arm_a], arms[arm_b]
        raw = score_b.raw.csi - score_a.raw.csi
        calibrated = score_b.calibrated.csi - score_a.calibrated.csi
        dev_a, dev_b = score_a.raw.deviation, score_b.raw.deviation
        gap = None if dev_a is None or dev_b is None else dev_a - dev_b
        pairs.append(PairAudit(
            arm_a=arm_a,
            arm_b=arm_b,
            raw_contrast=raw,
            calibrated_contrast=calibrated,
            distortion=raw - calibrated,
            calibration_gap=gap,
            ranking_reversal=(raw > 0) != (calibrated > 0),
        ))

    normalized_pooling = {"avg": "mean", "average": "mean",
                          "identity": "none"}.get(pooling, pooling)
    return AuditReport(
        threshold=float(threshold),
        pooling=normalized_pooling,
        block_size=int(block_size),
        arms=arms,
        pairs=pairs,
        calibrators=calibrators,
    )
