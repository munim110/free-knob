# Out-of-sample test of the bias-gap relation

g = |log B_a| − |log B_b|; D = raw − calibrated margin (exact post-pooling threshold).

**Fit on the six SEVIR arms** (450 contrasts): D = +0.0015 + +0.0864·g, residual sd 0.0176, r 0.896, ρ 0.860. Gap range in the fit [-1.74, +1.71]. Cells with |g| < 0.05: n = 114, mean |D| = 0.0035.

| held-out system | n | g range | ρ(g, D) | sign agreement, abs(g) > 0.1 | inside fitted range: n, within 95% band | outside range: n | mean abs error |
|---|---|---|---|---|---|---|---|
| DiffCast (CVPR 2024): DiffCast, PhyDNet backbone, persistence | 84 | [-4.46, +4.18] | 0.924 | 97% (n = 65) | 59, 81% | 25 | 0.0641 |
| PreDiff (NeurIPS 2023) − persistence | 29 | [-0.32, +0.06] | 0.658 | 100% (n = 2) | 29, 100% | 0 | 0.0032 |

The linear form is fitted on gaps up to about 1.7; outside that range CSI's bounds make D saturate, so only rank and sign are tested there.
