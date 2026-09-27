"""Dataset adapter for the RainNet RY product (RYDL.hdf5).

Matches the RainNet training protocol:
    input  = stack(x[t-15], x[t-10], x[t-5], x[t])   -> 4 channels
    target = x[t+5]                                   -> 1 channel

Both input and target are transformed with ln(x + 0.01) and mirror-padded
from 900x900 -> 928x928 so that height/width are multiples of 16 (four 2x
pools in RainNet / RainNet-DualDecoder).

Temporal contiguity is enforced: each sample uses 5 consecutive keys whose
timestamps are exactly 5 minutes apart. Samples that straddle radar
outages or QC gaps are skipped.

Two split regimes:

  "default" (our earlier pilot):
      train: 2012, 2013, 2015
      val:   2016
      test:  2017

  "paper" (Ayzel 2020, approximated. 2014 is missing from RYDL.hdf5):
      train: 2006-2013
      val:   2015
      test:  2016, 2017
      filters: May-Sep only, and >=10% of the 900x900 domain with intensity
               >=0.125 mm/h (= 0.0104 mm/5min) at the forecast time t
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


RAW_SHAPE = 900
PADDED_SHAPE = 928
PAD = (PADDED_SHAPE - RAW_SHAPE) // 2  # 14
STEP_MINUTES = 5
SEQ_LEN = 5  # 4 inputs + 1 target
LOG_OFFSET = 0.01

DEFAULT_SPLITS = {
    # 2014 is missing from RYDL.hdf5; 2013 is sparse but included. Train set
    # spans 2012, 2013, 2015 (~200k frames). Chronological: older -> newer.
    "train": ("2012", "2013", "2015"),
    "val":   ("2016",),
    "test":  ("2017",),
}

# Ayzel 2020 paper's splits (2014 missing from RYDL.hdf5).
PAPER_SPLITS = {
    "train": ("2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013"),
    "val":   ("2015",),
    "test":  ("2016", "2017"),
}

WET_THRESHOLD_MM_PER_5MIN = 0.125 / 12.0   # paper's 0.125 mm/h
WET_MIN_FRAC = 0.10                         # paper's 10% domain coverage
SUMMER_MONTHS = (5, 6, 7, 8, 9)             # paper's May-Sep window


def _parse_timestamp(key: str) -> dt.datetime:
    """Parse a RYDL key like '201701010000' into a datetime."""
    return dt.datetime.strptime(key, "%Y%m%d%H%M")


def _find_contiguous_starts(
    keys: Sequence[str],
    seq_len: int = SEQ_LEN,
    step_minutes: int = STEP_MINUTES,
) -> list[int]:
    """Return indices i such that keys[i..i+seq_len-1] are consecutive in time.

    Runs in O(n), no datetime parsing inside the loop after the first pass.
    """
    stamps = [_parse_timestamp(k) for k in keys]
    step = dt.timedelta(minutes=step_minutes)
    starts: list[int] = []
    run = 1
    for i in range(1, len(stamps)):
        run = run + 1 if (stamps[i] - stamps[i - 1]) == step else 1
        if run >= seq_len:
            starts.append(i - seq_len + 1)
    return starts


def _mirror_pad(a: np.ndarray) -> np.ndarray:
    return np.pad(a, ((PAD, PAD), (PAD, PAD)), mode="reflect")


def _log_transform(a: np.ndarray) -> np.ndarray:
    return np.log(a.astype(np.float32) + LOG_OFFSET)


def log_to_mm(x: np.ndarray | torch.Tensor):
    """Inverse of the ln(x + 0.01) transform."""
    import torch as _t
    if isinstance(x, _t.Tensor):
        return _t.exp(x) - LOG_OFFSET
    return np.exp(x) - LOG_OFFSET


def mm_to_log(mm: float) -> float:
    return float(np.log(mm + LOG_OFFSET))


class RainNetRYDataset(Dataset):
    """Streaming dataset over 5-frame contiguous sequences from RYDL.hdf5.

    Each __getitem__ returns:
        predictor:    (4, 928, 928) float32, log-transformed, mirror-padded
        target_log:   (1, 928, 928) float32, log-transformed, mirror-padded
        target_mm:    (1, 928, 928) float32, raw mm/5min (unpadded region valid,
                                              padding is mirrored mm too)

    The third entry matches the `MultiVariableARDataset` 3-tuple convention so
    the same training loop code can iterate over both.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "train",
        years: Iterable[str] | None = None,
        max_samples: int | None = None,
        subset_seed: int | None = None,
        split_regime: str = "default",
        summer_only: bool = False,
        wet_filter: bool = False,
        wet_min_frac: float = WET_MIN_FRAC,
        wet_coverage_path: str | Path | None = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        if years is None:
            splits = PAPER_SPLITS if split_regime == "paper" else DEFAULT_SPLITS
            if split not in splits:
                raise ValueError(f"Unknown split {split!r}; pass `years=` explicitly")
            years = splits[split]
        self.years = tuple(years)
        self.split = split
        self.split_regime = split_regime
        self.summer_only = summer_only
        self.wet_filter = wet_filter
        self.wet_min_frac = wet_min_frac

        # One-time index build (keys list + contiguous-start list)
        with h5py.File(self.hdf5_path, "r") as h:
            all_keys = sorted(h.keys())

        self._keys = [k for k in all_keys if k[:4] in self.years]

        # May-Sep filter: drop keys outside the summer months entirely. This
        # also breaks temporal contiguity across the filter boundary, which
        # is what we want (we don't stitch Apr -> May sequences).
        if self.summer_only:
            self._keys = [k for k in self._keys if int(k[4:6]) in SUMMER_MONTHS]

        self._starts = _find_contiguous_starts(self._keys)

        # Wet-domain filter: drop sequences whose FORECAST-TIME frame (t, the
        # 4th of 5) has <10% wet coverage. Uses a precomputed cache keyed by
        # HDF5 key so we don't rescan 800k frames each run.
        if self.wet_filter:
            if wet_coverage_path is None:
                wet_coverage_path = self.hdf5_path.parent / "wet_coverage.npz"
            cache = np.load(wet_coverage_path, allow_pickle=False)
            cache_keys = cache["keys"].astype(str).tolist()
            cache_frac = cache["frac"]
            key_to_frac = dict(zip(cache_keys, cache_frac.tolist()))
            # index 3 in the 5-frame window = t (the last input / forecast time)
            kept = [s for s in self._starts
                    if key_to_frac.get(self._keys[s + 3], 0.0) >= self.wet_min_frac]
            self._starts = kept

        if max_samples is not None and max_samples < len(self._starts):
            if subset_seed is not None:
                rng = np.random.default_rng(subset_seed)
                idx = rng.permutation(len(self._starts))[:max_samples]
                idx.sort()  # keep chronological order inside the subset
                self._starts = [self._starts[i] for i in idx]
            else:
                self._starts = self._starts[:max_samples]

        # Late-bind the h5 handle so it's forked-safe across workers
        self._h5 = None

    def __len__(self) -> int:
        return len(self._starts)

    def _h(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._h5

    def __getitem__(self, idx: int):
        start = self._starts[idx]
        h = self._h()

        frames = []
        for j in range(SEQ_LEN):
            a = np.asarray(h[self._keys[start + j]], dtype=np.float32)
            frames.append(_mirror_pad(a))

        predictor = np.stack([_log_transform(f) for f in frames[:4]], axis=0)  # (4, 928, 928)
        target_mm_2d = frames[4]                                               # (928, 928)
        target_log = _log_transform(target_mm_2d)[None, ...]                   # (1, 928, 928)
        target_mm = target_mm_2d[None, ...].astype(np.float32)                 # (1, 928, 928)

        return (
            torch.from_numpy(predictor),
            torch.from_numpy(target_log),
            torch.from_numpy(target_mm),
        )

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import paths as P
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else P.RADAR_H5
    for split in ("train", "val", "test"):
        ds = RainNetRYDataset(path, split=split)
        print(f"[{split:5s}] {len(ds):>6,d} contiguous 5-frame sequences "
              f"over years {ds.years}")
    ds = RainNetRYDataset(path, split="val", max_samples=2)
    x, ylog, ymm = ds[0]
    print(f"predictor:  {tuple(x.shape)}   dtype={x.dtype}   min={x.min():.3f}  max={x.max():.3f}")
    print(f"target_log: {tuple(ylog.shape)} dtype={ylog.dtype} min={ylog.min():.3f} max={ylog.max():.3f}")
    print(f"target_mm:  {tuple(ymm.shape)}  dtype={ymm.dtype}  min={ymm.min():.3f}  max={ymm.max():.3f}")
    print(f"log-to-mm sanity: mm_to_log(1.0)={mm_to_log(1.0):.4f}, mm_to_log(5.0)={mm_to_log(5.0):.4f}")
