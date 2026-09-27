"""Positive fraction of the SEVIR evaluation half at each threshold.

Quoted in the introduction, so it is computed rather than estimated: the full
evaluation half (odd event indices), all 12 lead times, unpooled.
"""
import json
from pathlib import Path

import h5py
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

THRESHOLDS = [16, 74, 133, 160, 181, 219]
GT = Path(P.SEVIR_H5)

f = h5py.File(GT, "r")["OUT_vil"]
idx = np.arange(1, f.shape[0], 2)
tot, pos = 0, {t: 0 for t in THRESHOLDS}
for s in range(0, len(idx), 16):
    g = f[np.sort(idx[s:s + 16])].astype(np.float32)
    tot += g.size
    for t in THRESHOLDS:
        pos[t] += int((g >= t).sum())
out = {"n_events": int(len(idx)), "n_pixels": int(tot),
       "R": {str(t): pos[t] / tot for t in THRESHOLDS}}
Path(str(P.RESULTS / "sevir_base_rates.json")).write_text(
    json.dumps(out, indent=2))
for t in THRESHOLDS:
    print(f"tau={t:>3}  R={100*out['R'][str(t)]:.4f}%  "
          f"1 in {1/max(out['R'][str(t)],1e-12):.0f}")
