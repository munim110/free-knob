# Getting started

This repository is self-contained for checking every reported numerical result from
cached artefacts. Raw datasets, model checkpoints, and prediction HDF5 files are
not redistributed because of their size and upstream terms.

## Fastest checks

1. Install `numpy` and `matplotlib` with `pip install -r requirements.txt`.
2. Run `python tools/verify.py` from the repository root.
3. Inspect `results/freeknob_parity_cascast_det.json` for the full SEVIR parity
   record and `MANIFEST.md` for result-level provenance.

The verification script runs 28 fast tests (26 for the audit package and 2 for
the atmospheric base rates), checks every result hash, regenerates all
numerical LaTeX fragments and figures in a temporary directory, and compares
every numerical fragment byte-for-byte, ignoring line endings, with
`reference_outputs/`. It does not need a GPU, dataset, or checkpoint.

## FreeKnob Audit

The installable audit protocol is the smallest independently useful component:

```bash
python -m pip install -e .
python -m freeknob --help
python -m unittest discover -s tests -p "test_*.py"
```

`freeknob.free_knob_audit` applies one matched calibration instrument to every
arm using a caller-supplied calibration split, evaluates the fixed instruments
on a disjoint split, and reports frequency bias plus raw and calibrated pairwise
contrasts. The real-data parity harness is CPU-only but requires the external
SEVIR test HDF5 and cached prediction HDF5 documented in `REPRODUCE.md`.

## Full reproduction

`REPRODUCE.md` distinguishes three levels:

- Tier 1: rebuild paper quantities from the included cached artefacts.
- Tier 2: rerun analyses from included intermediate artefacts.
- Tier 3: regenerate intermediates from public datasets and released model
  checkpoints; this level requires external downloads and GPUs.

The dual-decoder network used in the additional matched-control runs (radar,
infrared and crowd density) is named `DualDecoder` in the code and in result
file names.
