"""Score CasCast's cached coarse latents, decoded, on CasCast's own test split.

Their evaluation never re-runs the deterministic backbone: it reads a cached
`encode(EarthFormer(x)).sample()` per sample. Decoding those latents therefore
gives their backbone's field as their cascade actually sees it -- a VAE round
trip of it, which is exactly the quantity their own `latent_compress_model`
scores when it builds the cache.

This is the backbone half of the split question, and it costs one decoder pass
rather than a diffusion run. If the backbone cell barely moves between our split
and theirs, the split cannot be carrying the cascade's gap either.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402

REPO = Path(str(P.BASELINES / "cascast_repo"))
sys.path.insert(0, str(REPO))
from networks.autoencoder_kl import autoencoder_kl        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascast_their_split import (SCALE_FACTOR, VIL_SCALE, THRESHOLDS,  # noqa: E402
                                 POOLINGS, pool, as_cascast)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", type=Path,
                    default=P.BASELINES / "cascast_latents" /
                    "earthformer_example" / "test_2h")
    ap.add_argument("--list", type=Path,
                    default=REPO / "datasets/sevir_list/test.txt")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path(str(P.BASELINES / "cascast")))
    ap.add_argument("--cfg", type=Path,
                    default=REPO / "configs/sevir_used/cascast_diffusion.yaml")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(open(args.cfg, "r"))
    ae = autoencoder_kl(OmegaConf.to_object(
        cfg.model.params.sub_model.autoencoder_kl)).to(device).eval()
    sd = torch.load(args.ckpt_dir / "cascast_casformer.pth",
                    map_location="cpu", weights_only=False)["model"]["autoencoder_kl"]
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("state_dict mismatch; refusing to run")

    names = [l.strip() for l in args.list.read_text().splitlines() if l.strip()]
    fg = h5py.File(args.gt, "r")
    G = fg["OUT_vil"]
    N = len(names) if not args.limit else min(args.limit, len(names))
    print(f"decoding {N} cached coarse latents")

    CONV = ("ours", "cascast")
    hits = {c: {p: {t: np.zeros(N, np.int64) for t in THRESHOLDS}
                for p in POOLINGS} for c in CONV}
    fas = {c: {p: {t: np.zeros(N, np.int64) for t in THRESHOLDS}
               for p in POOLINGS} for c in CONV}
    miss = {c: {p: {t: np.zeros(N, np.int64) for t in THRESHOLDS}
                for p in POOLINGS} for c in CONV}
    bs = args.batch_size
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            z = torch.from_numpy(np.stack([
                np.load(args.latent_dir / names[i]) for i in range(s, e)
            ])).to(device).float()
            b, t = z.shape[0], z.shape[1]
            # the cache is unscaled; their decode_stage divides by the scale
            # factor, so scaling then decoding round-trips to the same field
            dec = ae.net.decode(z.reshape(b * t, *z.shape[2:]))
            pred = dec.reshape(b, t, 384, 384).clamp(0, 1) * VIL_SCALE
            gt = torch.from_numpy(
                G[s:e].astype(np.float32)).to(device).permute(0, 3, 1, 2)
            for conv in CONV:
                x, y = ((pred, gt) if conv == "ours"
                        else (as_cascast(pred), as_cascast(gt)))
                for pl in POOLINGS:
                    pp, gg = pool(x, pl), pool(y, pl)
                    for th in THRESHOLDS:
                        ph, gh = pp >= th, gg >= th
                        d = (1, 2, 3)
                        hits[conv][pl][th][s:e] = torch.count_nonzero(
                            ph & gh, dim=d).cpu()
                        fas[conv][pl][th][s:e] = torch.count_nonzero(
                            ph & ~gh, dim=d).cpu()
                        miss[conv][pl][th][s:e] = torch.count_nonzero(
                            ~ph & gh, dim=d).cpu()
            if (s // bs) % 100 == 0:
                print(f"  {e}/{N}", flush=True)
    fg.close()

    table = []
    for conv in CONV:
        for pl in POOLINGS:
            for th in THRESHOLDS:
                H, Fa, M = (int(hits[conv][pl][th].sum()),
                            int(fas[conv][pl][th].sum()),
                            int(miss[conv][pl][th].sum()))
                table.append({"convention": conv, "pooling": pl,
                              "threshold": th,
                              "csi": H / max(H + Fa + M, 1),
                              "bias": (H + Fa) / max(H + M, 1),
                              "hits": H, "fas": Fa, "miss": M})
    args.out.write_text(json.dumps({
        "arm": "cascast_cached_coarse_decoded",
        "split": "cascast_earthformer_test", "n": N,
        "table": table,
        "per_sample": {c: {p: {str(t): {"hits": hits[c][p][t].tolist(),
                                        "fas": fas[c][p][t].tolist(),
                                        "miss": miss[c][p][t].tolist()}
                               for t in THRESHOLDS} for p in POOLINGS}
                       for c in CONV},
        "complete": True}))
    pub = {("CSI-M", "none"): 0.4310, ("CSI-M", "max16"): 0.4351,
           ("CSI-219", "none"): 0.1448, ("CSI-219", "max16"): 0.1481}
    print(f"\n{'metric':>8} {'pool':>6} {'published':>10} "
          f"{'ours conv':>10} {'their conv':>11}")
    for (metric, pl), pv in pub.items():
        row = {}
        for c in CONV:
            sel = [r for r in table if r["convention"] == c and r["pooling"] == pl]
            row[c] = (sum(r["csi"] for r in sel) / 6 if metric == "CSI-M"
                      else [r for r in sel if r["threshold"] == 219][0]["csi"])
        print(f"{metric:>8} {pl:>6} {pv:10.4f} "
              f"{row['ours']:10.4f} {row['cascast']:11.4f}   "
              f"({(row['cascast'] - pv) / pv:+.1%} in their units)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
