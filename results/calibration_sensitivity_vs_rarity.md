# What a free monotone recalibration buys, versus event rarity

Attention U-Net under a weighted MSE on geostationary infrared band B08. The event is `y <= tau` K, so rarity rises as tau falls. Ten runs per threshold: five seeds by two single-decoder loss variants. `identity` is the model's own output; `recal` is the knob chosen on validation by pooled bias -> 1. Each threshold was retrained at its own tau, so R is the only quantity that moves.

`n` counts the runs entering the relative column, which is undefined for a run at CSI 0; `n_all` counts every run, and the absolute and median columns use all of them.

| tau | R (test) | CSI identity | CSI recal | dCSI | rel gain | n | n_all | dCSI (all) | median dCSI (all) |
|---|---|---|---|---|---|---|---|---|---|
| 220 | 20.270% | 0.2672 | 0.2610 | -0.0062 | -2.3% | 10 | 10 | -0.0062 | -0.0048 |
| 215 | 11.537% | 0.1992 | 0.1956 | -0.0036 | -1.8% | 10 | 10 | -0.0036 | -0.0011 |
| 210 | 6.378% | 0.1381 | 0.1368 | -0.0013 | -0.9% | 10 | 10 | -0.0013 | -0.0039 |
| 205 | 3.124% | 0.0578 | 0.0739 | +0.0160 | +27.7% | 10 | 10 | +0.0160 | +0.0143 |
| 200 | 0.884% | 0.0117 | 0.0352 | +0.0236 | +202.1% | 9 | 10 | +0.0238 | +0.0245 |
| 197 | 0.264% | 0.0085 | 0.0299 | +0.0214 | +250.8% | 7 | 10 | +0.0237 | +0.0265 |

Spearman(relative gain, log10 R) = -1.000, p = 0.0028

Range: -2.3% at the most common threshold to +250.8% at the rarest.

## The same sweep within each loss variant

The pooled table above averages the five seeds of both single-decoder loss variants. Averaging can manufacture an ordering, so here is the same statistic computed inside each variant separately.

| variant | tau 220 | tau 215 | tau 210 | tau 205 | tau 200 | tau 197 | Spearman vs log10 R | two-sided p |
|---|---|---|---|---|---|---|---|---|
| graded | -1.2% (5/5) | -2.5% (5/5) | +2.3% (5/5) | +47.5% (5/5) | +277.3% (4/5) | +451.2% (3/5) | -0.943 | 0.0167 |
| tiered | -3.5% (5/5) | -1.2% (5/5) | -4.2% (5/5) | +11.4% (5/5) | +164.6% (5/5) | +189.0% (4/5) | -0.829 | 0.0583 |

Cells give the relative gain with the count entering it out of the runs trained at that threshold. The p is the same exact two-sided permutation test used on the pooled row, so all three are comparable. Six levels give the test almost no resolution: the smallest two-sided p attainable is 2/720, and only a near-perfect ordering clears a conventional threshold, which is why the claim rests on the size of the gains and where they sit rather than on this test.

## Runs excluded from the relative column

Each is listed with its uncalibrated and recalibrated CSI, so the table is self-contained and the exclusion can be checked here rather than against a working file.

- tau 200: `rsweep_B08_t200_graded_s42` scores 0.0000 uncalibrated and 0.0258 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
- tau 197: `rsweep_B08_t197_graded_s13` scores 0.0000 uncalibrated and 0.0265 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
- tau 197: `rsweep_B08_t197_graded_s42` scores 0.0000 uncalibrated and 0.0297 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
- tau 197: `rsweep_B08_t197_tiered_s0` scores 0.0000 uncalibrated and 0.0315 recalibrated, so its relative gain is unbounded and it is excluded from that column only.
