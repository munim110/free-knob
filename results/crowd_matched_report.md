# Crowd domain: matched-knob comparison

Both sides receive {gain, shift, qmap+gain}; the operating point is chosen on validation by |bias - 1|. `beta only` is the old asymmetric comparison, in which the dual decoder got no post-hoc knob.

| threshold | R | DualDecoder matched | best single | margin | seeds | p | margin (beta only) |
|---|---|---|---|---|---|---|---|
| 2.0 | 16.82% | 0.4218 | 0.4684 | -0.0465 | 2/5 | 0.312 | -0.0813 |
| 3.0 | 11.26% | 0.4223 | 0.4466 | -0.0244 | 2/5 | 0.625 | -0.0286 |
| 4.0 | 8.67% | 0.4366 | 0.4428 | -0.0062 | 2/5 | 0.812 | -0.0329 |
| 5.0 | 6.28% | 0.3955 | 0.3910 | +0.0045 | 3/5 | 1.000 | -0.0421 |
| 7.0 | 3.77% | 0.3553 | 0.3207 | +0.0346 | 5/5 | 0.062 | -0.0489 |
| 10.0 | 2.50% | 0.3038 | 0.2841 | +0.0197 | 4/5 | 0.312 | -0.0743 |
| 15.0 | 1.82% | 0.3541 | 0.2968 | +0.0574 | 5/5 | 0.062 | -0.1495 |
