# A Free Knob: A Sensitivity Audit for Calibration in Threshold-Based Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2609.33457-b31b1b.svg)](https://arxiv.org/abs/2609.33457)

Code and result artefacts for the paper by Md Tanveer Hossain Munim, Bijoy Ahmed
Saiem, Al-Amin Sany and Tanzima Hashem
([arXiv:2609.33457](https://arxiv.org/abs/2609.33457)). Every number, table and
figure in the paper is generated from the files in this repository by the scripts
in this repository; none is typed by hand.

## What the paper claims

A large family of dense-prediction benchmarks scores models by thresholding
prediction and target at a rare value, reducing both over spatial blocks, and
computing a skill score over the contingency table. At rare thresholds the score
is dominated by how far each model's output sits from calibration rather than by
where it places things, and that component is removable by a monotone,
label-free, post-hoc transform that cannot reorder any two pixels.

The consequence is that a comparison which does not grant that transform to every
arm confounds architecture with calibration. Across 450 contrasts between six
SEVIR systems, the calibration gap between two arms predicts how far their
reported difference moves once both are calibrated (r = +0.884 at the reference
cell), and 51 of the 450 contrasts reverse sign. Both figures carry event-level
clustered bootstrap intervals, because the 450 cells share arms and share events,
and both hold when the transform is refitted on a validation window that precedes
the test period rather than on a half of it.

Under average pooling, a map applied to pixels before pooling can reorder blocks.
The exact control fits the map after pooling instead. It is then one threshold
per arm under every pooling and cannot reorder any block. Under it the gap
predicts the movement with r = +0.896, and 64 of the 450 contrasts reverse sign,
46 of them with both margins outside their 95% intervals.

The audit is repeated on four comparisons between systems released by their own
authors: DiffCast against its PhyDNet backbone and PreDiff against persistence on
SEVIR, five global models on WeatherBench 2, and AIFS against IFS on ECMWF open
data. PreDiff is the predicted null case. Both of its arms sit near unit bias,
and the control moves no cell by more than 0.020 CSI. A line fitted on the six
SEVIR arms alone predicts the DiffCast and PreDiff movement out of sample.

## FreeKnob Audit: the named check

The quantile map itself is classical. **FreeKnob Audit** names the protocol this
paper contributes: fit that same instrument separately to every arm on a shared,
disjoint calibration split; score every fixed map on the evaluation split; and
report pooled frequency bias as the one-number diagnostic. For a pair `(A, B)`,
the report also gives the signed calibration gap
`|log Bias(A)| - |log Bias(B)|`, which predicts how the raw contrast moves in
the paper's experiments.

Install the reference implementation from the repository root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .       # Windows
# .venv/bin/python -m pip install -e .         # Linux/macOS
```

The public function implements the boxed protocol in Section 8:

```python
from freeknob import free_knob_audit

report = free_knob_audit(
    {"baseline": baseline_cal, "candidate": candidate_cal},
    observations_cal,
    {"baseline": baseline_test, "candidate": candidate_test},
    observations_test,
    threshold=219,
    pooling="max",
    block_size=16,
)

print(report.arms["candidate"].raw.frequency_bias)
print(report.pairs[0].calibration_gap)
print(report.pairs[0].raw_contrast, report.pairs[0].calibrated_contrast)
```

All arrays have the same shape within a split. Pooling reduces non-overlapping
blocks over the final two dimensions before thresholding, matching the paper.
The caller supplies the calibration/test split so leakage remains visible at the
call site. See `python -m freeknob --help` for the NPZ command-line interface.

By default the map is fitted on pixels and applied before pooling, as in the
paper's main control. Pass `control="exact"` to fit it after pooling instead.
Each arm then gets one forecast threshold on pooled block values, chosen so that
its exceedance count on the calibration split equals the observed count. The
thresholds are returned in `report.forecast_thresholds`. On the command line the
same option is `--control exact`.

For datasets too large to hold in memory, `spatial_pool`, `contingency_score`,
`threshold_pool_score` and `exact_pooled_threshold` expose the same tested
primitives for chunked accumulation. The high-level function remains the recommended interface when the
arrays fit in memory.

### Real-data parity test

The slow parity harness checks the public implementation against all 30 raw and
global-quantile-map cells in a shipped SEVIR control. It needs the official
SEVIR nowcast test HDF5 and the corresponding cached prediction HDF5, neither of
which is included in this repository because of size.

```bash
python -m pip install -e ".[parity]"
python tests/sevir_parity.py \
  --pred ../results/sevir_pred_cascast_det.h5 \
  --gt ../data/sevir/nowcast_testing_000.h5 \
  --gold results/sevir_poolingq_cascast_det.json \
  --out results/freeknob_parity_cascast_det.json
```

The harness fits the same 1,001-quantile global map from the same calibration
events and accumulates evaluation counts in bounded HDF5 batches. The released
result, `results/freeknob_parity_cascast_det.json`, passes all 30 cells at an
absolute tolerance of `1e-7` on CPU. The fast unit/API/CLI suite contains 37
tests and requires no data download.

## Quick start

The following CPU-only check runs the 37 fast tests, regenerates every numerical
LaTeX output and all four figures, compares the numerical outputs byte-for-byte
(ignoring line endings) with the shipped references, and validates all result
hashes:

```bash
python -m venv .venv
# Windows: .venv/Scripts/python -m pip install -r requirements.txt
# Linux/macOS: .venv/bin/python -m pip install -r requirements.txt
python tools/verify.py
```

See `GETTING_STARTED.md` for the shortest routes through the package and
`REPRODUCE.md` for all three reproduction tiers.

## Layout

```
repository root/
  freeknob/        installable FreeKnob Audit reference implementation
  tests/           protocol, sign-convention and command-line tests
  sevir/           SEVIR nowcasting: inference, pooling sweep, decomposition,
                   pairwise contrasts, two bootstraps, base rates, rarity
                   sweep, the validation-window refit, the exact control, and
                   the DiffCast and PreDiff audits
  crowd/           ShanghaiTech Part A under its own patch-sum protocol,
                   with the dataset and model code vendored under shanghaitech/,
                   and the exact control
  nwp/             24-h precipitation audits of WeatherBench 2 and ECMWF
                   open-data forecasts
  segmentation/    COCO val2017 with a released DeepLabV3, where IoU is
                   algebraically the same statistic
  atmospheric/     AR event selection, dataset construction, calibration
                   control and training for the infrared rarity sweep
  radar/           additional matched-control run on radar (not in the paper)
  selection/       operating-point selection rules for the matched controls
  common/          dataset, model and loss definitions shared by the above
  paper/           generate numbers.tex, tab_*.tex and the figures
  tools/           path resolution, statistics, manifest generation, verification
  results/         released artefacts, both the ones the paper reads and the
                   intermediates that feed them; results/revision/ holds the
                   exact control, the reruns and the independent systems
  reference_outputs/ generated LaTeX and figures used for verification
  licenses/        license text for redistributed third-party-derived code
  GETTING_STARTED.md shortest guided route through the package
  THIRD_PARTY_NOTICES.md provenance and redistribution notices
  MANIFEST.md      every artefact with sha256, producer and consumer
  REPRODUCE.md     exact commands, in order
  pyproject.toml   package metadata and the freeknob-audit entry point
  requirements.txt dependencies for the CPU-only verification tier
```

Nothing outside this repository is required beyond the public datasets and the
released checkpoints listed in REPRODUCE.md. All dataset, model and loss code the
scripts import is vendored under `common/` and `crowd/shanghaitech/`.

## Three reproduction tiers

**Tier 1 rebuilds every number in the paper from the cached artefacts in
`results/`.** It needs Python and about ten seconds. No GPU, no dataset, no
checkpoints. This tier is the one to run first, because it verifies that the
released code and the released artefacts produce the released manuscript.

```
python paper/make_numbers.py
python paper/make_figures.py
```

**Tier 2 reruns the analyses from the intermediate artefacts.** Every report in
the paper can be regenerated without retraining anything, because the calibration
controls those reports consume are shipped in `results/`. This is the tier most
readers will want.

**Tier 3 regenerates everything from raw data and released checkpoints.** It
needs the SEVIR nowcast test split, ShanghaiTech Part A, COCO val2017, the RYDL
radar archive, the preprocessed atmospheric fields, the EarthFormer, CasCast,
DiffCast and PreDiff checkpoints, and a GPU. The precipitation audits read public
WeatherBench 2 and ECMWF buckets and need no GPU. REPRODUCE.md gives the commands
and the cost of each stage.

## Paths

Scripts read their locations from `tools/paths.py`, which takes defaults from the
repository layout and allows any of them to be overridden by an environment
variable:

```
SEVIR_DATA     directory holding nowcast_testing_000.h5, and the training
               split if the validation-window control is to be rerun
RESULTS_DIR    where artefacts are read and written
PAPER_DIR      where numbers.tex, tab_*.tex and figures are written
               (default: generated/ inside this directory)
BASELINES_DIR  released checkpoints and upstream repositories
CROWD_DIR      ShanghaiTech Part A working copy, if you have your own
COCO_DIR       COCO val2017 images and annotations
ATMOS_DATA     preprocessed atmospheric fields, one directory per band
RADAR_H5       RYDL.hdf5 radar archive
RUNS           training run directories holding model checkpoints
```

A command-line flag always wins over the environment variable, which wins over
the default.

## Requirements

```
pip install -r requirements.txt
```

Tier 1 needs only `numpy` and `matplotlib`. Tiers 2 and 3 additionally need
`torch`, `h5py`, `scipy`, `omegaconf`, `pycocotools`, `joblib` and `pysteps`,
plus the upstream EarthFormer and CasCast repositories on disk for Tier 3. The
DiffCast and PreDiff runs need their authors' repositories. The precipitation
audits in `nwp/` need `xarray`, `zarr`, `gcsfs`, `pandas`, `eccodes` and
`numcodecs`.

## Provenance of the models we evaluate

Every arm comes from a released artefact rather than a reimplementation.

| arm | source |
|---|---|
| persistence | last observed frame of the input sequence |
| pysteps | Lucas-Kanade advection, `pysteps` library |
| EarthFormer | authors' released `earthformer_sevir.pt`; the rerun uses PaddleScience's release of the same weights |
| CasCast backbone | CasCast's released deterministic checkpoint |
| CasCast cascade | CasCast's released cascade, cfg=2 as their eval script runs it |
| DiffCast and its PhyDNet backbone | authors' released `diffcast_phydnet_sevir128.pt`; one sampling call returns both |
| PreDiff | authors' SEVIR-LR weights, recovered from their repository |
| GraphCast, FuXi, IFS HRES, IFS ENS mean | WeatherBench 2 public forecasts, 2020 |
| AIFS Single, AIFS ENS, IFS HRES, IFS ENS | ECMWF open data, 2025-08-28 to 2026-05-11 |
| DeepLabV3-ResNet50 | `torchvision` released checkpoint |

Two reproduction anchors are reported in the paper. The CasCast backbone matches
all four published cells to within 2.8%, which is what licenses decomposing the
authors' own numbers. The segmentation pipeline reproduces the published argmax
mIoU for its checkpoint, with two named protocol differences we do not adopt.

The official EarthFormer checkpoint URL now returns an error, so the rerun loads
PaddleScience's copy of the same weights, converted by
`sevir/convert_paddle_earthformer.py`. It matches the released EarthFormer
scores to 2.7e-4 CSI.

PreDiff's original weight URLs no longer resolve. We recovered the weights from
the authors' repository and scored 2,027 test windows with one sample each.
Pooled CSI is within 3% of the published value and 12% below it at pixel level, a
gap we have not resolved. The audit's conclusion does not turn on that offset:
both arms sit near unit frequency bias, so the control moves the margin very
little either way.

DGMR and NowcastNet are absent for a different reason. Their released checkpoints
are fitted to other radar products, UK composite radar and US MRMS, at different
resolutions and time steps from SEVIR's 384x384 VIL, so putting either on this
benchmark means retraining, and a retrained model is a reimplementation. Nothing
in the claim depends on which models are present: the diagnostic costs one pooled
bias number, so anyone holding a SEVIR checkpoint we do not can run the same
check.

## Citation

If you use this code or the FreeKnob Audit, please cite the paper:

```bibtex
@misc{munim2026freeknob,
  title         = {A Free Knob: A Sensitivity Audit for Calibration
                   in Threshold-Based Evaluation},
  author        = {Munim, Md Tanveer Hossain and Saiem, Bijoy Ahmed and
                   Sany, Al-Amin and Hashem, Tanzima},
  year          = {2026},
  eprint        = {2609.33457},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2609.33457},
  url           = {https://arxiv.org/abs/2609.33457}
}
```

## License

The code and result artefacts are released under the MIT License (see `LICENSE`).
Third-party components and their terms are listed in `THIRD_PARTY_NOTICES.md`.
