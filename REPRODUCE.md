# Reproducing the paper

Three tiers. Tier 1 rebuilds every reported number from the cached artefacts and
takes seconds. Tier 2 reruns each analysis from the shipped intermediate
artefacts, so no training is needed. Tier 3 regenerates everything from raw data
and released checkpoints, and is where the cost is.

All commands are run from the repository root.

The matched-versus-asymmetric comparisons of a dual-decoder network (radar,
infrared and crowd density; `tab_radar.tex`, `tab_atmos.tex`, `tab_crowd.tex`)
are additional analyses. They are reproducible from this directory but are not
reported in the manuscript.

---

## Install and verify the named audit

The paper's boxed check is available as the `free_knob_audit` function and the
`freeknob-audit` command. The calibration instrument is classical quantile
mapping; the named contribution is the matched-arm, disjoint-split protocol and
its pooled-frequency-bias diagnostic.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m unittest discover -s tests -v
.venv/Scripts/freeknob-audit --help
```

On Linux or macOS, replace `.venv/Scripts/` with `.venv/bin/`. The command-line
input is one NPZ containing `obs_calibration`, `obs_test`, and a pair of arrays
named `ARM__calibration`, `ARM__test` for every arm. For example:

```bash
freeknob-audit predictions.npz --threshold 219 --pooling max --block-size 16 \
    --output audit.json
```

The JSON records raw and calibrated counts, CSI, frequency bias, per-arm gain,
and every pair's raw contrast, calibrated contrast, distortion, signed
calibration gap, and ranking-reversal flag.

The fast suite is complemented by a real-data parity harness. After placing the
official SEVIR test HDF5 and cached deterministic prediction at the paths below:

```bash
.venv/Scripts/python -m pip install -e ".[parity]"
.venv/Scripts/python tests/sevir_parity.py \
    --pred ../results/sevir_pred_cascast_det.h5 \
    --gt ../data/sevir/nowcast_testing_000.h5 \
    --gold results/sevir_poolingq_cascast_det.json \
    --out results/freeknob_parity_cascast_det.json
```

This is CPU-only. It reads evaluation events in bounded batches and checks all
five pooling conventions at all six thresholds against the shipped paper
artefact. The shipped parity record passes all 30 cells at absolute tolerance
`1e-7`; the accompanying fast unit/API/CLI suite has 26 tests.

---

## Tier 1: artefacts to manuscript

Needs Python, `numpy`, `matplotlib`. No GPU, no dataset, no checkpoints.

```
python paper/make_numbers.py
python paper/make_figures.py
```

This reads the artefacts in `results/` and writes `numbers.tex`, `tab_*.tex` and
the two figures into `PAPER_DIR`. It prints one summary line per experiment and
reports any macro the manuscript references but no artefact supports.

Expected output includes:

```
  E1: 6 rarity levels, rho=-1.000
  E6: gain range -8.8% (avg16) to +79.7% (max16)
  dose: rho=0.675 over 30 arm-threshold points at max16, 0.599 over all 150 cells
  E8: reported +109.1%, residual after global qmap +12.9% (share 88%)
  repro: worst deviation 2.8%
  contrasts: 450 cells over 15 pairs, r=+0.884 (max16), 51 flips (23 substantive)
  crowdpool: rho=0.568 over 320 trained cells, 5 seeds
  segpool: 20 classes / 5000 images; rho(rarity)=-0.019 PREDICTION FAILED
  ets: erased share at 219 is 88% under ETS against 88% under CSI; largest gap
       over thresholds 5 points
  bootstrap: cascade+cal - det+cal at 219 = +0.0340 [+0.0234, +0.0439]
  wrote numbers.tex with 284 macros
```

Every rank correlation above is Spearman's rho with tied ranks averaged
(`tools/stats.py`). An earlier version ranked with `argsort(argsort(x))`, which
assigns ordinal ranks and breaks ties by input order; on the segmentation table,
where several per-class gains are identically zero, that made the reported value
depend on the order the classes happened to arrive in, over a range of about
0.11. Six correlations move slightly against manuscripts built before this fix,
none of them enough to change a claim.

A run that prints `PLACEHOLDERS still outstanding` has an artefact missing.
Compare `results/` against MANIFEST.md.

To confirm the artefacts are the ones the paper was built from:

```
python tools/make_manifest.py
git diff --exit-code MANIFEST.md
```

Then build the manuscript:

```
cd ../paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Tier 2: intermediates to reports

No GPU and no training. The calibration controls these consume are shipped, so
each report is reproducible on a laptop. Each command below regenerates a file in
`results/` that should come back byte-identical.

```
python sevir/published_contrasts.py --results results

python sevir/decomposition_report.py --det cascast_det --cascade cascast_cascade

python sevir/rarity_sweep.py \
    --controls results/rsweep_control_matched_t*.json \
    --r-test results/rsweep_R_table.json \
    --out results/calibration_sensitivity_vs_rarity.md

python selection/protocol_selection_report.py \
    --controls results/rsweep_control_matched_t*.json \
    --r-clim results/R_clim_train_B08.json \
    --r-test results/rsweep_R_table.json \
    --key csi --out results/selection_matched_csi.md

python crowd/matched_report.py \
    --control results/posthoc_crowd_matched.json \
    --out results/crowd_matched_report.md

python crowd/pooling_report.py

python radar/matched_report.py \
    --dual_decoder results/posthoc_radar_dual_decoder_matched_s42.json \
    --single results/posthoc_radar_plain_mse_s42.json \
             results/posthoc_radar_plain_mse_s43.json \
             results/posthoc_radar_plain_mse_s44.json \
    --out results/radar_matched_report.md

python sevir/contrast_bootstrap.py --n-boot 2000
```

The last one reuses the shipped per-event count cache, so it needs no prediction
HDF5s and takes seconds rather than the half hour the cache itself costs to
build. All eight have been verified to reproduce their shipped artefacts exactly.

---

## Tier 3: raw data to artefacts

### Inputs

| what | where it goes | size |
|---|---|---|
| SEVIR nowcast test split, `nowcast_testing_000.h5` | `SEVIR_DATA` | 8 GB |
| EarthFormer checkpoint `earthformer_sevir.pt` | `BASELINES_DIR` | 35 MB |
| EarthFormer repository | `BASELINES_DIR/ef` | |
| CasCast repository and checkpoints | `BASELINES_DIR/cascast_repo`, `BASELINES_DIR/cascast` | |
| ShanghaiTech Part A | `CROWD_DIR` | 500 MB |
| COCO val2017 and instance annotations | `COCO_DIR` | 6 GB |
| RYDL radar archive `RYDL.hdf5` | `RADAR_H5` | |
| preprocessed atmospheric fields, per band | `ATMOS_DATA` | |
| SEVIR nowcast training split and its META csv | `SEVIR_DATA` | 30 GB |

The training split is needed only for the validation-window control of Stage 2c.
Both it and the test split come from `s3://sevir/data/processed/` with
`aws s3 cp --no-sign-request` and no credentials.

The dataset, model and loss definitions these scripts import are vendored under
`common/` and `crowd/shanghaitech/`, so no external working copy is needed. If
you have your own ShanghaiTech tree, point `CROWD_DIR` at it and it takes
precedence over the vendored copy.

COCO can be fetched with `scripts/get_coco.sh` in the parent repository.

### Stage 1: SEVIR predictions

Each writes one cached prediction file so the calibration control can be applied
post hoc without re-running a network per knob setting.

```
python sevir/persistence_baseline.py --out results/sevir_pred_persistence.h5

python sevir/infer_earthformer.py --out results/sevir_pred_earthformer.h5

python sevir/infer_earthformer.py \
    --ckpt $BASELINES_DIR/cascast/cascast_deterministic.pth \
    --ckpt-path model.EarthFormer_xy --strip-prefix net. \
    --out results/sevir_pred_cascast_det.h5

python sevir/infer_cascast.py --coarse results/sevir_pred_cascast_det.h5 \
    --cfg-weight 2 --out results/sevir_pred_cascast_cascade.h5
python sevir/infer_cascast.py --coarse results/sevir_pred_cascast_det.h5 \
    --cfg-weight 1 --out results/sevir_pred_cascast_cascade_cfg1.h5

python sevir/pysteps_baseline.py --out results/sevir_pred_pysteps.h5
```

CasCast's deterministic backbone is EarthFormer's cuboid transformer, so
`infer_earthformer.py` builds it and the two flags above pull their checkpoint
out of the nested state dict CasCast releases. There is no separate script for
it, and the fact that one script loads both is the same fact
Section 6.2 of the paper reports.

Two settings matter and are easy to get wrong. Predictions are handled in VIL
units with the repositories' own `x/255` convention. The CasCast cascade is
sampled with classifier-free guidance weight 2 and a single ensemble member,
matching the authors' released `scripts/eval_diffusion_infer.sh`; the unguided
path undershoots their reported CSI-219-POOL16 by about 13%.

`pysteps_baseline.py` requires OpenCV for Lucas-Kanade. Without it the library
falls back to persistence on every event while still producing plausible numbers,
so the script counts fallbacks and warns above 1%.

Cost: a few GPU-hours, dominated by the 20-step DDIM sampling of the cascade.

### Stage 2: pooling sweep

One invocation per arm. Persistence is derived from the input sequence and needs
no prediction file.

```
for arm in persistence pysteps earthformer cascast_det cascast_cascade cascast_cascade_cfg1; do
  python sevir/pooling_control.py \
      --pred results/sevir_pred_$arm.h5 \
      --name $arm \
      --out results/sevir_poolingq_$arm.json
done
```

Cost: minutes per arm on CPU.

### Stage 2b: per-event counts, and the panel of Figure 1

```
python sevir/contrast_bootstrap.py --n-boot 2000 \
    --cache results/sevir_perevent_counts.npz \
    --out results/sevir_contrast_bootstrap.json

python sevir/make_schematic.py --pred results/sevir_pred_cascast_det.h5 \
    --out results/sevir_schematic.npz
```

The first caches per-event contingency counts for every arm under the identity
knob, its own global quantile map, and, when the Stage 2c artefacts are already
present, the prior-period map as a third column. Building the cache takes about
half an hour and dominates; the 2000 replicates on top of it are seconds, so
rerunning the bootstrap with a different seed reuses the cache.

The shipped caches (`sevir_perevent_counts.npz`, `sevir_bootstrap_cascast.json`,
`sevir_val_fitted.json`) were built from an earlier inference run of the
deterministic CasCast backbone than the one behind
`sevir_poolingq_cascast_det.json`. The two runs differ by at most 1.1e-4 CSI over
the 30 cells; the other five arms are identical. So that one value appears
throughout the paper, `make_numbers.py` takes the identity and test-half point
estimates in the bootstrap and prior-window tables from the pooling-control
record (`canon()`), and takes only the intervals and the prior-window column
from the caches. Rebuilding the caches from the current prediction file removes
the difference.

Run Stage 2c *before* this one if you can. Reading and pooling the prediction
fields is identical work for every knob, so caching all three columns in one pass
costs nothing extra, and it is what lets the val-fit intervals be computed later
with numpy alone:

```
python sevir/valfit_bootstrap.py
```

That script needs no HDF5, no torch and no GPU. It reads the cache, resamples the
same event indices with the same seed and replicate count as Appendix L, and
writes `sevir_valfit_bootstrap.json` with the residual interval under each fit.
If the cache has only two knob columns it says so and exits; delete it and rerun
`contrast_bootstrap.py` with the `sevir_valpred_*.h5` caches and the val subset
in place. Keep the replicate count at 2000: matching Appendix L is what makes the
two tables comparable, and raising it changes nothing a reader can act on.

The second caches the fields behind Figure 1(a). Both artefacts are shipped, so
Tier 1 rebuilds the figure and the intervals without them.

### Stage 2c: the knob refitted on a prior-period validation split

```
python sevir/make_val_subset.py --n-events 128

V=$SEVIR_DATA/nowcast_val_subset.h5
python sevir/persistence_baseline.py --h5 $V --out results/sevir_valpred_persistence.h5
python sevir/infer_earthformer.py    --h5 $V --out results/sevir_valpred_earthformer.h5
python sevir/infer_earthformer.py    --h5 $V \
    --ckpt $BASELINES_DIR/cascast/cascast_deterministic.pth \
    --ckpt-path model.EarthFormer_xy --strip-prefix net. \
    --out results/sevir_valpred_cascast_det.h5
python sevir/infer_cascast.py --coarse results/sevir_valpred_cascast_det.h5 \
    --cfg-weight 2 --out results/sevir_valpred_cascast_cascade.h5
python sevir/infer_cascast.py --coarse results/sevir_valpred_cascast_det.h5 \
    --cfg-weight 1 --out results/sevir_valpred_cascast_cascade_cfg1.h5
python sevir/pysteps_baseline.py --h5 $V --out results/sevir_valpred_pysteps.h5

python sevir/val_fitted_control.py
```

`make_val_subset.py` needs `nowcast_training_000.h5` and its META csv, both from
`s3://sevir/data/processed/` with no credentials. It selects the window both
EarthFormer and CasCast hold out from training, 2019-01-01 to 2019-04-30, which
also ends before the test period starts.

### Stage 2d: the map fitted per lead time

Appendix M. Fits one quantile map per forecast horizon instead of one for the
whole field, under two calibration budgets, and rescores the reference cell.

```
for spec in cascast_det:cascast_det \
            cascade_tiled_s0:cascade_tiled_s0 \
            cascade_indep_s0:cascade_indep_s0; do
  name="${spec%%:*}"
  for b in matched equal; do
    python sevir/leadtime_qmap.py --pred results/sevir_pred_$name.h5 \
        --name $name --budget $b \
        --out results/leadtime/${name}_${b}.json
  done
done
```

Cost: about six minutes per run under `matched`, nearer ten under `equal`, which
reads more calibration events to fit each horizon's map. Needs the Stage 1
prediction files; nothing else depends on this stage, so it can be skipped and
Appendix M's macros will show as placeholders.

The two budgets are the point. Under `matched` every map is fitted from the same
calibration events, so a per-horizon map sees a twelfth of the pixels; under
`equal` each is given as many pixels as the global map. The global map is fitted
from a fixed event sample in both, so its column is identical across them --- if
a rerun moves the global column between budgets, the event sampling has drifted
and the comparison is void.

### Stage 2e: knob family against selection rule

The family-against-rule appendix, and the SEVIR half of Appendix J. Accumulates a
lead-resolved contingency table for all 25 members of the discrete knob family on
both halves, then reads four rules off it in one pass.

```
for name in cascast_det cascade_tiled_s0 cascade_indep_s0; do
  python sevir/knob_selection.py --pred results/sevir_pred_$name.h5 \
      --name $name --out results/knobrule/${name}.json
done
```

Cost: about twelve minutes per arm. Restricted to `--poolings none,max16` by
default, since the appendix quotes the reference cell; passing all five poolings
multiplies the cost by about 2.5 and changes no number the paper reports.

Both selection rules see the calibration half only, so neither is an oracle; the
`csi_oracle` column is the best evaluation-half score over the family and is
printed to bound what the rules leave behind, not as a result. The check that the
harness is sound is that pooled-horizon selection by bias reproduces the per-cell
knob of Section 6 to four figures --- if it does not, the knob family or the
calibration split has drifted.

### Stage 3: SEVIR analysis

```
python sevir/published_contrasts.py --results results
python sevir/decomposition_report.py --det cascast_det --cascade cascast_cascade
python sevir/base_rates.py
python sevir/bootstrap.py \
    --arm det=results/sevir_pred_cascast_det.h5:identity \
    --arm det+cal=results/sevir_pred_cascast_det.h5:qmap_global \
    --arm cascade=results/sevir_pred_cascast_cascade.h5:identity \
    --arm cascade+cal=results/sevir_pred_cascast_cascade.h5:qmap_global \
    --n-boot 2000 --seed 0 --out results/sevir_bootstrap_cascast.json
```

The bootstrap resamples events, not pixels, and uses one shared event-index
vector per replicate across all arms, so the interval on a difference reflects
the between-arm correlation. Only the evaluation half is resampled; the knob is
fixed once on the calibration half and does not enter the interval.

The same four arms under the prior-period fit of Appendix H, which puts an
interval on the residual a deployed system would see. Needs Stage 6's
`sevir_valpred_*.h5` caches and the val subset:

```
python sevir/bootstrap.py \
    --arm det=results/sevir_pred_cascast_det.h5:identity \
    --arm det+val=results/sevir_pred_cascast_det.h5:qmap_val \
    --arm cascade=results/sevir_pred_cascast_cascade.h5:identity \
    --arm cascade+val=results/sevir_pred_cascast_cascade.h5:qmap_val \
    --n-boot 2000 --seed 0 --out results/sevir_bootstrap_valfit.json
```

Cost: seconds, except each bootstrap at roughly ten minutes for 2000 replicates.

### Stage 4: rarity sweep

The training threshold is set equal to the evaluation threshold and the model
retrained at each level, so the positive fraction is the only quantity that
moves. Six levels, ten seeds each.

Training produces the checkpoints; `atmospheric/calibration_control.py` in
Stage 7 turns them into the control JSONs; the Tier 2 command then produces the
report. Cost: this is the most expensive stage, since it retrains at every level.

### Stage 5: crowd density

Five seeds, two trained arms, plus the constructed control arms that move
placement and sharpness one at a time.

```
for s in 42 43 44 45 46; do
  python crowd/pooling_control.py --arms baseline,csrnet --seed $s \
      --out results/crowd_pooling_s$s.json
done
python crowd/pooling_report.py
```

The matched report needs the crowd calibration control from Stage 7.

Pooling is applied to the grid of 32x32 patch counts, so `none` reproduces
ShanghaiTech's published protocol exactly and the thresholds stay in the
benchmark's own units.

Cost: about two minutes per seed on one GPU.

### Stage 6: segmentation

```
python segmentation/repro_check.py
python segmentation/pooling_control.py --arch deeplabv3_resnet50 --size 520 \
    --out results/seg_pooling.json
```

`repro_check.py` scores argmax mIoU the way the checkpoint's authors do, which is
the anchor. `pooling_control.py` then thresholds the class probability, which is
what provides an operating point to move. Only contingency counts are stored, so
memory stays constant across 5000 images and 15 knob settings.

Cost: about 20 minutes on one GPU for the 5000-image sweep.

### Stage 7: the calibration controls

These are the stages that need trained checkpoints, and they produce the
intermediate artefacts Tier 2 consumes.

#### Building the atmospheric dataset

The atmospheric setting is the one domain whose training recipe we control, so
its inputs are built here rather than downloaded ready made. Both sources are
public: ERA5 for the predictor fields, and the geostationary B08 water-vapour
channel for the target. Event selection comes from a published global
atmospheric-river catalogue computed on ERA5, so what counts as an event is
fixed by a community algorithm.

```
python atmospheric/select_ar_events.py \
    --catalog $ATMOS_DATA/globalARcatalog_ERA5_v4.nc \
    --out $ATMOS_DATA/ar_event_times.txt
```

That writes every 6-hourly catalogue step carrying an AR object inside the study
box. The catalogue query spans 2010 to 2023, while the usable record is its
intersection with target availability: 1,500 selected scenes from 2015-07-17
06:00 through 2023-12-17 06:00. Pair each retained step into
`<stamp>_predictor.npy` of shape `(5, 351, 251)` holding IVT, T500, T850, RH700
and W500 interpolated to the 0.02-degree target grid, and
`<stamp>_target.npy` holding B08 brightness temperature. The scored models use
T500, T850, RH700 and W500; IVT is stored but not passed to them. Then:

```
python atmospheric/prepare_dataset.py \
    --processed-dir $ATMOS_DATA/processed_B08 --band 8
```

This splits 80/10/10 by selected-scene timestamp and writes
`$ATMOS_DATA/B08/{train,val,test}` beside `normalization_stats.joblib`, the
layout `--data-dir` expects below. The exact spans are 1,200 training scenes
(2015-07-17 06:00--2021-11-14 18:00), 150 validation scenes
(2021-11-15 06:00--2023-03-19 06:00), and 150 test scenes
(2023-03-19 12:00--2023-12-17 06:00). A random split would leak between nearby
selected scenes.

Predictor means and population standard deviations are accumulated separately
per channel over training pixels. The archived statistics used by the scored
checkpoints have SHA-256
`d50471e7e03847d93ca293660a51934203156b09df499c4a9a8b42c76bde52b6`;
their target mean and standard deviation are 229.3998806 K and 11.1880560 K.
The included preparation script implements that estimator.

Audit every atmospheric base-rate definition from the targets alone:

```
python atmospheric/base_rates.py --data-dir $ATMOS_DATA/B08 \
    --out results/atmospheric_base_rates.json \
    --r-test-out results/rsweep_R_table.json \
    --r-clim-out results/R_clim_train_B08.json
```

The paper's `R` is the all-frame test fraction on the valid 256x251 evaluation
footprint inside the nominal 256x256 tensor. Rule D instead uses the training
fraction on that footprint after the historical strict coverage >0.1% filter.
At 220 K these are 20.2697% and 26.0067%, respectively; the all-frame full-grid
test fraction is 19.6215%.

#### Training the checkpoints

Every flag below is read off the `train_config.json` written beside the released
checkpoints, so these are the runs the paper scores rather than a plausible
reconstruction. Seeds are `{0, 7, 13, 21, 42}` throughout.

Atmospheric single decoders, both loss variants, at each threshold of the rarity
sweep. `--loss-thresholds` places the second tier at the temperature whose
training positive fraction is a third of the first tier's, which is the rule the
sweep applies at every level rather than a per-level choice:

```
python atmospheric/train_unet.py \
    --data-dir $ATMOS_DATA/B08 --output-dir $RUNS/unet_B08_s$SEED --band 8 \
    --epochs 50 --batch-size 8 --learning-rate 1e-4 \
    --early-stopping-patience 10 --gradient-clip 1.0 --seed $SEED \
    --variables T500,T850,RH700,W500 --base-channels 64 --unet-depth 4 \
    --loss-thresholds '{"220.0": 10.0, "210.0": 25.0}' \
    --sampling-weight-threshold 220.0 --amp
```

The graded variant is the same call against `common/losses/graded.py`. The dual
decoder holds `--beta 2.0`, the published value, at every threshold:

```
python atmospheric/train_dual_decoder.py \
    --data-dir $ATMOS_DATA/B08 --output-dir $RUNS/dual_decoder_B08_s$SEED --band 8 \
    --epochs 50 --batch-size 8 --learning-rate 1e-4 \
    --early-stopping-patience 10 --gradient-clip 1.0 --seed $SEED \
    --variables T500,T850,RH700,W500 \
    --critical-threshold-k 220.0 --critical-mode low --beta 2.0 \
    --bg-weight 1.0 --ext-weight 1.0 --combined-weight 1.0 \
    --lambda-weight 10.0 --use-weighted-sampler \
    --sampling-weight-threshold 220.0 --val-csi-threshold-k 220.0 --amp
```

Radar, three single-decoder seeds and one dual:

```
python radar/train_rainnet.py --hdf $RADAR_H5 \
    --output-dir $RUNS/rainnet_plain_mse_s$SEED --loss mse \
    --epochs 10 --batch-size 8 --learning-rate 1e-4 \
    --early-stopping-patience 3 --gradient-clip 10.0 --seed $SEED \
    --critical-threshold-mm 0.5 --split-regime paper \
    --summer-only --wet-filter --wet-min-frac 0.1 --amp

python radar/train_rainnet_dual_decoder.py --hdf $RADAR_H5 \
    --output-dir $RUNS/rainnet_dual_decoder_s$SEED --arch dual_decoder \
    --epochs 10 --batch-size 8 --learning-rate 1e-4 \
    --early-stopping-patience 3 --gradient-clip 10.0 --seed $SEED \
    --critical-threshold-mm 0.5 --critical-mode high --beta 2.0 \
    --bg-weight 1.0 --ext-weight 1.0 --combined-weight 1.0 \
    --lambda-weight 100.0 --loss-bg logcosh --loss-cmb mse \
    --split-regime paper --summer-only --wet-filter --wet-min-frac 0.1 --amp
```

Radar training is the expensive stage in this section, about 70 minutes per epoch
on one GPU, so roughly 12 GPU-hours per seed. This analysis uses one dual-decoder
seed here and says so.

Crowd checkpoints come from `crowd/shanghaitech/train.py`, five seeds per arm.

#### The controls

```
python atmospheric/calibration_control.py \
    --data-dir $ATMOS_DATA/B08 --band 8 --threshold-k 220 --filter-percent 0.1 \
    --unet-glob "$RUNS/rsweep_B08_t220_*_s*/attention_unet*B08.pth" \
    --dual_decoder-glob "$RUNS/rsweep_B08_t220_dual_decoder_s*/dual_decoder_unified_B08_best_csi.pth" \
    --out results/rsweep_control_matched_t220.json

python radar/calibration_control.py --model rainnet --checkpoint <ckpt> \
    --out results/posthoc_radar_plain_mse_s42.json

python crowd/calibration_control.py --outputs-dir <crowd runs> --seeds 42,43,44,45,46 \
    --out results/posthoc_crowd_matched.json
```

Repeat the atmospheric command for each threshold in
{220, 215, 210, 205, 200, 197}. Then rerun the Tier 2 reports.

---

## Determinism

The calibration and evaluation halves are split by index parity, so no stored
seed is needed to reproduce the split. SEVIR events split 936 each; COCO splits
on even and odd image index.

The bootstrap takes `--seed`. Crowd training takes `--seed` and the paper reports
all five rather than a selected one.

Diffusion sampling is stochastic. We evaluate one sample per event, matching the
released evaluation script, so cascade numbers move slightly between runs. The
calibration asymmetry the decomposition isolates does not depend on the sample.
