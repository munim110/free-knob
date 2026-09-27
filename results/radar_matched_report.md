# Radar (DWD RY / RainNet): matched vs asymmetric control

`dual (beta only)` is the original protocol: the dual decoder may move only its architectural blend weight. `dual (matched)` gives it the same {gain, qmap} family the single decoder always had. The single-decoder column is the mean over three seeds, each selected by the same rule.

| threshold | dual (beta only) | dual (matched) | single (matched) | margin, beta only | margin, matched |
|---|---|---|---|---|---|
| 0.1mm | 0.7543 | 0.7543 | 0.7553 | -0.0009 | -0.0009 |
| 0.5mm | 0.5899 | 0.5897 | 0.5873 | +0.0026 | +0.0025 |
| 1mm | 0.3891 | 0.4595 | 0.4532 | -0.0640 | +0.0063 |
| 2.5mm | 0.3354 | 0.3435 | 0.3440 | -0.0087 | -0.0005 |
| 5mm | 0.2058 | 0.2034 | 0.2110 | -0.0052 | -0.0076 |

The differential carries the result. The same two checkpoints, the same test set and the same selection rule are held fixed. Only whether both sides may use the knob family changes, and the margin moves with it.
