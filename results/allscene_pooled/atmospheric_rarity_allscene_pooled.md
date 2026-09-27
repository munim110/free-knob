# What a free monotone recalibration buys, versus event rarity (pooled CSI)

Attention U-Net under a weighted MSE on geostationary infrared band B08. The event is `y <= tau` K, so rarity rises as tau falls. Ten runs per threshold: five seeds by two single-decoder loss variants. `identity` is the model's own output; `recal` is the knob chosen on validation by pooled bias -> 1. Each threshold was retrained at its own tau.

Every validation and test scene enters one pooled contingency table; there is no event-coverage frame filter.

`n` counts the runs entering the relative column, which is undefined for a run at CSI 0; `n_all` counts every run, and the absolute and median columns use all of them.

| tau | R (test) | pooled CSI identity | pooled CSI recal | dpooled CSI | rel gain | n | n_all | dCSI (all) | median dCSI (all) |
|---|---|---|---|---|---|---|---|---|---|
| 220 | 20.270% | 0.2976 | 0.3324 | +0.0348 | +11.7% | 10 | 10 | +0.0348 | +0.0325 |
| 215 | 11.537% | 0.2302 | 0.2595 | +0.0294 | +12.8% | 10 | 10 | +0.0294 | +0.0305 |
| 210 | 6.378% | 0.1864 | 0.1919 | +0.0055 | +2.9% | 10 | 10 | +0.0055 | +0.0031 |
| 205 | 3.124% | 0.1140 | 0.1288 | +0.0148 | +13.0% | 10 | 10 | +0.0148 | +0.0069 |
| 200 | 0.884% | 0.0277 | 0.0533 | +0.0256 | +92.2% | 9 | 10 | +0.0271 | +0.0226 |
| 197 | 0.264% | 0.0175 | 0.0336 | +0.0161 | +92.2% | 7 | 10 | +0.0214 | +0.0277 |

Spearman(relative gain, log10 R) = -0.829, p = 0.0583

Range: +2.9% at the most common threshold to +92.2% at the rarest.

## The same sweep within each loss variant

The pooled table above averages the five seeds of both single-decoder loss variants. Averaging can manufacture an ordering, so here is the same statistic computed inside each variant separately.

| variant | tau 220 | tau 215 | tau 210 | tau 205 | tau 200 | tau 197 | Spearman vs log10 R | two-sided p |
|---|---|---|---|---|---|---|---|---|
| graded | +13.3% (5/5) | +10.9% (5/5) | +2.3% (5/5) | +23.9% (5/5) | +158.7% (4/5) | +156.7% (3/5) | -0.714 | 0.1361 |
| tiered | +10.1% (5/5) | +14.6% (5/5) | +3.6% (5/5) | +2.7% (5/5) | +59.9% (5/5) | +68.5% (4/5) | -0.486 | 0.3556 |

Cells give the relative gain with the count entering it out of the runs trained at that threshold. The p is the same exact two-sided permutation test used on the pooled row, so all three are comparable. Six levels give the test almost no resolution: the smallest two-sided p attainable is 2/720, and only a near-perfect ordering clears a conventional threshold, which is why the claim rests on the size of the gains and where they sit rather than on this test.

## Runs excluded from the relative column

Each is listed with its uncalibrated and recalibrated CSI, so the table is self-contained and the exclusion can be checked here rather than against a working file.

- tau 200: `rsweep_B08_t200_graded_s42` scores 0.0000 uncalibrated and 0.0410 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
- tau 197: `rsweep_B08_t197_graded_s13` scores 0.0000 uncalibrated and 0.0318 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
- tau 197: `rsweep_B08_t197_graded_s42` scores 0.0000 uncalibrated and 0.0404 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
- tau 197: `rsweep_B08_t197_tiered_s0` scores 0.0000 uncalibrated and 0.0292 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
