"""Resolve the directories the scripts read from and write to.

Every path has a default matching the layout in README.md and can be overridden
with the environment variable named beside it. Scripts take the values here as
argparse defaults, so a command-line flag still wins over both.

    SEVIR_DATA      directory holding nowcast_testing_000.h5
    RESULTS_DIR     where result JSON and markdown artefacts are written
    PAPER_DIR       where numbers.tex, tab_*.tex and the figures are written
    BASELINES_DIR   released checkpoints and upstream repositories
    CROWD_DIR       ShanghaiTech Part A working directory
    COCO_DIR        COCO val2017 images and annotations
    ATMOS_DATA      preprocessed atmospheric fields, one directory per band
    RADAR_H5        RYDL.hdf5 radar archive
    RUNS            training run directories holding model checkpoints

Example:

    RESULTS_DIR=/scratch/results python sevir/published_contrasts.py
"""
import os
from pathlib import Path

# tools/paths.py -> repository root
ARTIFACT = Path(__file__).resolve().parents[1]
REPO = ARTIFACT.parent


def _env(name, default):
    v = os.environ.get(name)
    return Path(v).expanduser() if v else Path(default)


SEVIR_DATA = _env("SEVIR_DATA", ARTIFACT / "data" / "sevir")
SEVIR_H5 = SEVIR_DATA / "nowcast_testing_000.h5"

RESULTS = _env("RESULTS_DIR", ARTIFACT / "results")
PAPER = _env("PAPER_DIR", ARTIFACT / "generated")

BASELINES = _env("BASELINES_DIR", Path.home() / "baselines")
CROWD = _env("CROWD_DIR", Path.home() / "shanghaitech_experiment")
COCO = _env("COCO_DIR", Path.home() / "coco")

ATMOS_DATA = _env("ATMOS_DATA", ARTIFACT / "data" / "processed")
RADAR_H5 = _env("RADAR_H5", ARTIFACT / "data" / "rainnet" / "RYDL.hdf5")
RUNS = _env("RUNS", ARTIFACT / "runs")


def require(path, what):
    """Fail with an actionable message rather than a stack trace deep in a loop."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"missing {what}: {p}\n"
            f"Set the matching environment variable (see tools/paths.py) "
            f"or pass the corresponding command-line flag.")
    return p
