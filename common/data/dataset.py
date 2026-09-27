"""Dataset for the AR downscaling task.

Loads (predictor, target) .npy pairs from a `B<band>/<split>/` directory and
normalizes both using the per-band stats joblib written by
atmospheric/prepare_dataset.py.

Predictor channels (in order):
    0: IVT
    1: T500
    2: T850
    3: RH700
    4: W500

The training script is responsible for slicing out the channel subset it wants.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


ALL_VARIABLES = ['IVT', 'T500', 'T850', 'RH700', 'W500']


class MultiVariableARDataset(Dataset):
    """Per-band downscaling dataset.

    Args:
        data_dir: root of the band's `B<band>/` directory, as written by
            atmospheric/prepare_dataset.py.
            Must contain `train/`, `val/`, `test/` subfolders and a
            `normalization_stats.joblib` file.
        split:   one of {"train", "val", "test"}.
        stats_filename: override for the stats file basename if needed.
    """

    def __init__(self, data_dir, split, stats_filename='normalization_stats.joblib'):
        data_dir = Path(data_dir)
        self.split_dir = data_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {self.split_dir}")

        self.predictor_files = sorted(self.split_dir.glob('*_predictor.npy'))
        if not self.predictor_files:
            raise FileNotFoundError(f"No predictor files found in {self.split_dir}")

        stats_path = data_dir / stats_filename
        if not stats_path.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")
        try:
            self.stats = joblib.load(stats_path)
        except (TypeError, KeyError) as exc:
            # joblib writes numpy arrays through a wrapper class whose pickled
            # state gained fields in 1.2. An older joblib fails on these files
            # inside the unpickler, several frames deep, with an error that says
            # nothing about versions.
            raise RuntimeError(
                f"could not read {stats_path} with joblib {joblib.__version__}: "
                f"{type(exc).__name__}: {exc}\n"
                f"These files were written by joblib >= 1.2 and older releases "
                f"cannot unpickle them. Install joblib>=1.2 (see "
                f"requirements.txt) and rerun.") from exc

        print(f"Loaded '{split}' dataset with {len(self.predictor_files)} samples from {self.split_dir}")

    def __len__(self):
        return len(self.predictor_files)

    def __getitem__(self, idx):
        pred_path = self.predictor_files[idx]
        targ_path = Path(str(pred_path).replace('_predictor.npy', '_target.npy'))
        case_name = pred_path.stem.replace('_predictor', '')

        full_predictor = np.load(pred_path).astype(np.float32)  # (C, H, W)
        target_data = np.load(targ_path).astype(np.float32)     # (H, W)

        # Cast back to float32, the stats are float64, so the broadcast
        # arithmetic above promotes the result, which would clash with AMP.
        predictor_norm = (
            (full_predictor - self.stats['predictor_mean'][:, None, None])
            / (self.stats['predictor_std'][:, None, None] + 1e-8)
        ).astype(np.float32)

        target_norm = (
            (target_data - self.stats['target_mean']) / (self.stats['target_std'] + 1e-8)
        ).astype(np.float32)

        return (
            torch.from_numpy(predictor_norm),
            torch.from_numpy(target_norm).unsqueeze(0),
            case_name,
        )


def resolve_variable_indices(variable_names):
    """Map a list of variable names to their channel indices in the saved npy files."""
    return [ALL_VARIABLES.index(v) for v in variable_names]
