# The knob refitted on a prior-period validation split

One global quantile map per arm, fitted on 128 events from 2019-01-01 to 2019-04-30 and applied unchanged to the 936 test evaluation events. `qmap_test` is the paper's default fit, on the test calibration half.

## Per arm at max 16x16 pooling, tau = 219

| arm | bias | CSI | CSI (test-fit) | CSI (val-fit) | gain test-fit | gain val-fit |
|---|---|---|---|---|---|---|
| persistence | 1.020 | 0.1924 | 0.1908 | 0.1936 | -0.8% | +0.6% |
| pysteps | 0.907 | 0.2700 | 0.2704 | 0.2719 | +0.2% | +0.7% |
| earthformer | 0.281 | 0.2082 | 0.2505 | 0.2877 | +20.3% | +38.2% |
| cascast_det | 0.172 | 0.1468 | 0.2638 | 0.2928 | +79.7% | +99.5% |
| cascast_cascade | 1.037 | 0.3069 | 0.2978 | 0.3063 | -2.9% | -0.2% |
| cascast_cascade_cfg1 | 0.548 | 0.2475 | 0.2899 | 0.2662 | +17.1% | +7.6% |

## The headline decomposition under each fit

| fit | raw | calibrated | erased |
|---|---|---|---|
| qmap_test | +109.1% | +12.9% | 88% |
| qmap_val | +109.1% | +4.6% | 96% |

## All pairwise contrasts under each fit

| fit | cells | Pearson(gap, distortion) | sign flips |
|---|---|---|---|
| qmap_test | 450 | +0.796 | 51 |
| qmap_val | 450 | +0.766 | 49 |
