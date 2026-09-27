# Published contrasts before and after calibrating both sides

`raw` is the contrast a table reports; `cal` is the same contrast with one global monotone quantile map fitted per arm on the calibration half and held fixed across every cell. Neither knob sees the evaluation half and neither can reorder pixels.


## Does the calibration gap predict the distortion?

| subset | n | Pearson | Spearman | sign flips | max abs distortion |
|---|---|---|---|---|---|
| all | 450 | +0.796 | +0.598 | 51 | 0.1261 |
| max16 | 90 | +0.884 | +0.787 | 17 | 0.1261 |
| extreme | 150 | +0.869 | +0.798 | 27 | 0.1261 |
| published | 90 | +0.791 | +0.660 | 17 | 0.1261 |

## Contrasts drawn from published comparisons, max16

| comparison | thr | bias A | bias B | raw | cal | rel raw | rel cal | flip |
|---|---|---|---|---|---|---|---|---|
| EarthFormer vs persistence | 16 | 0.993 | 0.915 | +0.0873 | +0.0304 | +12.3% | +4.3% |  |
| EarthFormer vs persistence | 74 | 0.996 | 0.747 | +0.0800 | +0.0825 | +13.3% | +13.7% |  |
| EarthFormer vs persistence | 133 | 0.999 | 0.534 | +0.0320 | +0.0567 | +7.3% | +13.1% |  |
| EarthFormer vs persistence | 160 | 0.993 | 0.387 | -0.0014 | +0.0565 | -0.4% | +16.9% | YES |
| EarthFormer vs persistence | 181 | 1.004 | 0.337 | +0.0066 | +0.0699 | +2.4% | +25.0% |  |
| EarthFormer vs persistence | 219 | 1.020 | 0.281 | +0.0158 | +0.0597 | +8.2% | +31.3% |  |
| CasCast cascade vs EarthFormer | 16 | 0.915 | 0.993 | -0.0126 | +0.0428 | -1.6% | +5.8% | YES |
| CasCast cascade vs EarthFormer | 74 | 0.747 | 0.961 | +0.0343 | +0.0316 | +5.0% | +4.6% |  |
| CasCast cascade vs EarthFormer | 133 | 0.534 | 0.905 | +0.0958 | +0.0661 | +20.5% | +13.5% |  |
| CasCast cascade vs EarthFormer | 160 | 0.387 | 0.875 | +0.1264 | +0.0642 | +38.0% | +16.4% |  |
| CasCast cascade vs EarthFormer | 181 | 0.337 | 0.904 | +0.1317 | +0.0641 | +46.0% | +18.3% |  |
| CasCast cascade vs EarthFormer | 219 | 0.281 | 1.037 | +0.0987 | +0.0473 | +47.4% | +18.9% |  |
| CasCast cascade vs CasCast backbone | 16 | 0.875 | 0.993 | -0.0017 | +0.0363 | -0.2% | +4.9% | YES |
| CasCast cascade vs CasCast backbone | 74 | 0.718 | 0.961 | +0.0492 | +0.0323 | +7.4% | +4.7% |  |
| CasCast cascade vs CasCast backbone | 133 | 0.495 | 0.905 | +0.1195 | +0.0607 | +26.9% | +12.2% |  |
| CasCast cascade vs CasCast backbone | 160 | 0.337 | 0.875 | +0.1579 | +0.0588 | +52.4% | +14.8% |  |
| CasCast cascade vs CasCast backbone | 181 | 0.285 | 0.904 | +0.1640 | +0.0523 | +64.7% | +14.5% |  |
| CasCast cascade vs CasCast backbone | 219 | 0.172 | 1.037 | +0.1601 | +0.0340 | +109.1% | +12.9% |  |

## One architecture, two trainings, max16

Not a published head-to-head. Both arms are the same cuboid transformer; one checkpoint is EarthFormer's release, the other is the CasCast authors' retraining of it as their deterministic backbone.

| thr | bias EF | bias CasCast-bb | raw | cal | rel raw | rel cal | flip |
|---|---|---|---|---|---|---|---|
| 16 | 0.915 | 0.875 | -0.0109 | +0.0065 | -1.4% | +0.9% | YES |
| 74 | 0.747 | 0.718 | -0.0149 | -0.0007 | -2.2% | -0.1% |  |
| 133 | 0.534 | 0.495 | -0.0236 | +0.0055 | -5.1% | +1.1% | YES |
| 160 | 0.387 | 0.337 | -0.0315 | +0.0054 | -9.5% | +1.4% | YES |
| 181 | 0.337 | 0.285 | -0.0324 | +0.0118 | -11.3% | +3.4% | YES |
| 219 | 0.281 | 0.172 | -0.0614 | +0.0133 | -29.5% | +5.3% | YES |

## Sources for the published pairs

- **CasCast cascade vs CasCast backbone**: CasCast Tab. 4/5, their own backbone-vs-cascade ablation
- **CasCast cascade vs EarthFormer**: CasCast Tab. 4, EarthFormer as a baseline it reports beating
- **EarthFormer vs persistence**: EarthFormer Tab. 3 reports persistence as a baseline

## Every sign flip, any pooling

| comparison | pooling | thr | raw | cal |
|---|---|---|---|---|
| EarthFormer vs persistence | max16 | 160 | -0.0014 | +0.0565 |
| CasCast backbone vs persistence | max16 | 160 | -0.0329 | +0.0620 |
| CasCast backbone vs persistence | max16 | 181 | -0.0258 | +0.0817 |
| CasCast backbone vs persistence | max16 | 219 | -0.0456 | +0.0730 |
| CasCast backbone vs pysteps (advection) | max4 | 219 | -0.0086 | +0.0783 |
| CasCast cascade (no guidance) vs pysteps (advection) | max16 | 16 | -0.0095 | +0.0081 |
| CasCast cascade (no guidance) vs pysteps (advection) | max16 | 74 | -0.0019 | +0.0041 |
| CasCast cascade (no guidance) vs pysteps (advection) | max16 | 133 | -0.0207 | +0.0033 |
| CasCast cascade (no guidance) vs pysteps (advection) | max16 | 160 | -0.0241 | +0.0101 |
| CasCast cascade (no guidance) vs pysteps (advection) | max16 | 181 | -0.0072 | +0.0251 |
| CasCast cascade (no guidance) vs pysteps (advection) | max16 | 219 | -0.0225 | +0.0195 |
| CasCast backbone vs EarthFormer | none | 133 | -0.0060 | +0.0018 |
| CasCast backbone vs EarthFormer | none | 160 | -0.0170 | +0.0016 |
| CasCast backbone vs EarthFormer | none | 181 | -0.0232 | +0.0025 |
| CasCast backbone vs EarthFormer | none | 219 | -0.0355 | +0.0092 |
| CasCast backbone vs EarthFormer | avg4 | 133 | -0.0030 | +0.0019 |
| CasCast backbone vs EarthFormer | avg4 | 160 | -0.0134 | +0.0025 |
| CasCast backbone vs EarthFormer | avg4 | 181 | -0.0200 | +0.0029 |
| CasCast backbone vs EarthFormer | avg4 | 219 | -0.0294 | +0.0110 |
| CasCast backbone vs EarthFormer | max4 | 74 | -0.0048 | +0.0027 |
| CasCast backbone vs EarthFormer | max4 | 133 | -0.0167 | +0.0036 |
| CasCast backbone vs EarthFormer | max4 | 160 | -0.0280 | +0.0014 |
| CasCast backbone vs EarthFormer | max4 | 181 | -0.0315 | +0.0043 |
| CasCast backbone vs EarthFormer | max4 | 219 | -0.0503 | +0.0089 |
| CasCast backbone vs EarthFormer | avg16 | 133 | +0.0070 | -0.0008 |
| CasCast backbone vs EarthFormer | max16 | 16 | -0.0109 | +0.0065 |
| CasCast backbone vs EarthFormer | max16 | 133 | -0.0236 | +0.0055 |
| CasCast backbone vs EarthFormer | max16 | 160 | -0.0315 | +0.0054 |
| CasCast backbone vs EarthFormer | max16 | 181 | -0.0324 | +0.0118 |
| CasCast backbone vs EarthFormer | max16 | 219 | -0.0614 | +0.0133 |
| CasCast cascade vs EarthFormer | max4 | 160 | +0.0128 | -0.0192 |
| CasCast cascade vs EarthFormer | max4 | 181 | +0.0200 | -0.0198 |
| CasCast cascade vs EarthFormer | max4 | 219 | +0.0102 | -0.0239 |
| CasCast cascade vs EarthFormer | avg16 | 160 | -0.0192 | +0.0318 |
| CasCast cascade vs EarthFormer | avg16 | 181 | -0.0118 | +0.0336 |
| CasCast cascade vs EarthFormer | max16 | 16 | -0.0126 | +0.0428 |
| CasCast cascade (no guidance) vs EarthFormer | avg16 | 181 | -0.0406 | +0.0039 |
| CasCast cascade (no guidance) vs EarthFormer | avg16 | 219 | -0.0391 | +0.0234 |
| CasCast cascade vs CasCast backbone | none | 219 | +0.0162 | -0.0466 |
| CasCast cascade vs CasCast backbone | avg4 | 219 | +0.0079 | -0.0414 |
| CasCast cascade vs CasCast backbone | max4 | 133 | +0.0128 | -0.0271 |
| CasCast cascade vs CasCast backbone | max4 | 160 | +0.0407 | -0.0206 |
| CasCast cascade vs CasCast backbone | max4 | 181 | +0.0515 | -0.0242 |
| CasCast cascade vs CasCast backbone | max4 | 219 | +0.0604 | -0.0329 |
| CasCast cascade vs CasCast backbone | avg16 | 160 | -0.0311 | +0.0228 |
| CasCast cascade vs CasCast backbone | avg16 | 181 | -0.0241 | +0.0192 |
| CasCast cascade vs CasCast backbone | avg16 | 219 | -0.0071 | +0.0064 |
| CasCast cascade vs CasCast backbone | max16 | 16 | -0.0017 | +0.0363 |
| CasCast cascade (no guidance) vs CasCast backbone | max4 | 181 | +0.0094 | -0.0416 |
| CasCast cascade (no guidance) vs CasCast backbone | max4 | 219 | +0.0168 | -0.0416 |
| CasCast cascade (no guidance) vs CasCast backbone | avg16 | 219 | -0.0532 | +0.0036 |
