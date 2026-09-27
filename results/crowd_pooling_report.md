# Crowd density: does the confound exist outside radar?

ShanghaiTech Part A, 182 test images, 32x32 patch sums, the benchmark's own protocol. Trained arms averaged over 5 seeds (42, 43, 44, 45, 46).

## Under the published protocol (`none`, a sum reduction)

CSI is the uncalibrated score; dCSI is the absolute change from the per-cell knob. The relative column is suppressed where the base CSI is below 0.02, since a ratio off a collapsed base is not a readable number.

| arm | tau | R | CSI | pooled Bias | dCSI | rel | seeds |
|---|---|---|---|---|---|---|---|
| baseline | 2 | 11.90% | 0.5010 $\pm$ 0.0387 | 1.225 $\pm$ 0.231 | +0.0049 $\pm$ 0.0091 | +1.1 $\pm$ 2.1% | 5 |
| baseline | 3 | 7.24% | 0.5075 $\pm$ 0.0355 | 1.000 $\pm$ 0.179 | +0.0021 $\pm$ 0.0107 | +0.4 $\pm$ 2.2% | 5 |
| baseline | 5 | 3.14% | 0.4496 $\pm$ 0.0436 | 0.994 $\pm$ 0.232 | -0.0159 $\pm$ 0.0319 | -4.0 $\pm$ 7.8% | 5 |
| baseline | 7 | 1.55% | 0.3806 $\pm$ 0.0539 | 1.128 $\pm$ 0.395 | -0.0339 $\pm$ 0.0288 | -10.0 $\pm$ 9.4% | 5 |
| baseline | 10 | 0.64% | 0.2847 $\pm$ 0.0729 | 1.426 $\pm$ 0.783 | -0.0442 $\pm$ 0.0482 | -21.0 $\pm$ 26.0% | 5 |
| baseline | 15 | 0.21% | 0.1953 $\pm$ 0.0928 | 1.998 $\pm$ 1.757 | -0.0614 $\pm$ 0.0388 | -48.5 $\pm$ 43.2% | 5 |

| csrnet | 2 | 11.90% | 0.4670 $\pm$ 0.0474 | 1.316 $\pm$ 0.276 | +0.0106 $\pm$ 0.0158 | +2.6 $\pm$ 4.1% | 5 |
| csrnet | 3 | 7.24% | 0.4632 $\pm$ 0.0367 | 1.122 $\pm$ 0.230 | +0.0072 $\pm$ 0.0108 | +1.7 $\pm$ 2.5% | 5 |
| csrnet | 5 | 3.14% | 0.3930 $\pm$ 0.0655 | 0.931 $\pm$ 0.296 | +0.0063 $\pm$ 0.0501 | +4.0 $\pm$ 16.8% | 5 |
| csrnet | 7 | 1.55% | 0.2853 $\pm$ 0.0881 | 0.682 $\pm$ 0.372 | +0.0352 $\pm$ 0.0729 | +28.0 $\pm$ 52.7% | 5 |
| csrnet | 10 | 0.64% | 0.1009 $\pm$ 0.1125 | 0.289 $\pm$ 0.431 | +0.0911 $\pm$ 0.1339 | +193.2 $\pm$ 231.1% | 5 |
| csrnet | 15 | 0.21% | 0.0333 $\pm$ 0.0667 | 0.103 $\pm$ 0.206 | -0.0203 $\pm$ 0.0694 | -93.8% | 5 |

| gt_shift2 | 2 | 11.90% | 0.2660 $\pm$ 0.0000 | 0.926 $\pm$ 0.000 | -0.0021 $\pm$ 0.0000 | -0.8 $\pm$ 0.0% | 5 |
| gt_shift2 | 3 | 7.24% | 0.2056 $\pm$ 0.0000 | 0.938 $\pm$ 0.000 | -0.0009 $\pm$ 0.0000 | -0.4 $\pm$ 0.0% | 5 |
| gt_shift2 | 5 | 3.14% | 0.1300 $\pm$ 0.0000 | 0.945 $\pm$ 0.000 | -0.0018 $\pm$ 0.0000 | -1.4 $\pm$ 0.0% | 5 |
| gt_shift2 | 7 | 1.55% | 0.0887 $\pm$ 0.0000 | 0.954 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_shift2 | 10 | 0.64% | 0.0432 $\pm$ 0.0000 | 0.969 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_shift2 | 15 | 0.21% | 0.0159 $\pm$ 0.0000 | 0.979 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | n/a | 5 |

| gt_blur1 | 2 | 11.90% | 0.9964 $\pm$ 0.0000 | 0.999 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur1 | 3 | 7.24% | 0.9936 $\pm$ 0.0000 | 0.998 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur1 | 5 | 3.14% | 0.9941 $\pm$ 0.0000 | 1.000 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur1 | 7 | 1.55% | 0.9910 $\pm$ 0.0000 | 1.001 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur1 | 10 | 0.64% | 0.9933 $\pm$ 0.0000 | 1.007 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur1 | 15 | 0.21% | 0.9948 $\pm$ 0.0000 | 0.995 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |

| gt_blur2 | 2 | 11.90% | 0.9884 $\pm$ 0.0000 | 0.999 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur2 | 3 | 7.24% | 0.9841 $\pm$ 0.0000 | 0.998 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur2 | 5 | 3.14% | 0.9747 $\pm$ 0.0000 | 0.995 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur2 | 7 | 1.55% | 0.9696 $\pm$ 0.0000 | 0.993 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur2 | 10 | 0.64% | 0.9667 $\pm$ 0.0000 | 1.003 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur2 | 15 | 0.21% | 0.9744 $\pm$ 0.0000 | 0.985 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |

| gt_blur4 | 2 | 11.90% | 0.9607 $\pm$ 0.0000 | 0.997 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur4 | 3 | 7.24% | 0.9549 $\pm$ 0.0000 | 0.995 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur4 | 5 | 3.14% | 0.9266 $\pm$ 0.0000 | 0.966 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur4 | 7 | 1.55% | 0.9291 $\pm$ 0.0000 | 0.975 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur4 | 10 | 0.64% | 0.9051 $\pm$ 0.0000 | 0.976 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur4 | 15 | 0.21% | 0.9082 $\pm$ 0.0000 | 0.928 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |

| gt_blur8 | 2 | 11.90% | 0.8951 $\pm$ 0.0000 | 0.992 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur8 | 3 | 7.24% | 0.8815 $\pm$ 0.0000 | 0.966 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur8 | 5 | 3.14% | 0.8483 $\pm$ 0.0000 | 0.940 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur8 | 7 | 1.55% | 0.8227 $\pm$ 0.0000 | 0.909 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur8 | 10 | 0.64% | 0.7967 $\pm$ 0.0000 | 0.876 $\pm$ 0.000 | +0.0000 $\pm$ 0.0000 | +0.0 $\pm$ 0.0% | 5 |
| gt_blur8 | 15 | 0.21% | 0.8030 $\pm$ 0.0000 | 0.887 $\pm$ 0.000 | +0.0124 $\pm$ 0.0000 | +1.5 $\pm$ 0.0% | 5 |

## Dose-response

Spearman(bias deficit, absolute dCSI from the knob), all usable cells over all seeds: **+0.559** over 1760 cells.

Restricted to under-forecasting cells (Bias < 1), the regime every SEVIR system occupies: **+0.555** over 1463 cells.

Over-forecasting cells (Bias >= 1, n=297) behave oppositely: correcting bias toward one *costs* CSI, because at a rare threshold scaling a too-massive field down removes hits faster than false alarms. Median gain there is +0.0000 CSI. The knob is not a free lunch in both directions, and we say so.

## Controls: sharpness and placement moved separately

| arm | what it holds fixed | Bias range | max abs gain |
|---|---|---|---|
| gt_shift2 | real field, placement wrong by 2 patches | [0.802, 1.036] | 0.0334 CSI |
| gt_blur8 | placement perfect, peaks destroyed | [0.803, 0.992] | 0.0999 CSI |

`gt_shift` is badly wrong and perfectly calibrated: a monotone knob cannot recover a displacement and correctly declines to act. `gt_blur` conserves mass, so a patch sum barely notices it. The metric punishes amplitude attenuation, and amplitude is what the knob gives back. Position and spread are untouched.

