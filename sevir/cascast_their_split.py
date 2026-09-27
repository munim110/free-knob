"""Reproduce CasCast's published CSI end to end on CasCast's own test split,
from CasCast's own cached conditioning latents.

Our main pipeline scores the cascade on the NeurIPS-2020 SEVIR nowcast
benchmark file (`nowcast_testing_000.h5`, 1872 curated events, one 25-frame
window each). CasCast does not: it evaluates on the Earthformer split, 12159
windows cut three per event at offsets 0/12/24 from 4053 raw 2019 events, listed
in `datasets/sevir_list/test.txt`. Their released archive
(`latent_data/earthformer_example`, named "example" but complete) holds the
coarse conditioning latent for every one of those 12159 windows, which is what
their evaluation actually consumes -- it never re-runs the deterministic
backbone.

So this script removes every remaining degree of freedom between us and their
number except the sampler and the decoder:

  * their split, in their order;
  * their cached coarse latents, so no backbone re-run and no redraw of the
    VAE posterior;
  * their sampler (DDIM 20 steps, cfg 2, noise tiled with period = eval batch
    size 8, generator seeded to 0 per batch);
  * ground truth cut from the raw SEVIR files by their documented rule;
  * our scorer, which was checked line by line against their
    `utils/metrics.py` and is arithmetically identical at these cells.

Predictions are scored as they are produced and never written to disk: at this
split size the cascade output would be ~70 GB and nothing downstream needs it.
Per-sample contingency counts are kept instead, which is all a pooled CSI or any
later resampling of it requires. The output JSON carries `complete`, and the
summary refuses to read a file that does not have it, so an interrupted run
cannot be mistaken for a result.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402

REPO = Path(str(P.BASELINES / "cascast_repo"))
sys.path.insert(0, str(REPO))

from networks.autoencoder_kl import autoencoder_kl        # noqa: E402
from networks.casformer import CasFormer                  # noqa: E402
from src.diffusers import DDIMScheduler                   # noqa: E402

SCALE_FACTOR = 0.6786020398139954
VIL_SCALE = 255.0
THRESHOLDS = [16, 74, 133, 160, 181, 219]
POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]


def build(device, cfg):
    p = cfg.model.params.sub_model
    ae = autoencoder_kl(OmegaConf.to_object(p.autoencoder_kl)).to(device).eval()
    cf = CasFormer(arch=p.casformer.arch,
                   config=OmegaConf.to_object(p.casformer.config)).to(device).eval()
    return ae, cf


def load_into(module, path, key_path, label):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for part in key_path.split("."):
        sd = sd[part]
    missing, unexpected = module.load_state_dict(sd, strict=False)
    print(f"  {label}: {len(sd)} tensors, missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    return len(missing), len(unexpected)


def as_cascast(x):
    """CasCast's own unit round trip, verbatim from SEVIRSkillScore.preprocess.

    Their dataset hands the metric a [0,1] tensor (`np.load(...) / 255`) and the
    metric converts it back with `x / (1./255.)`. Whether that is the identity
    depends on the device:

        CPU  : 120 of the 256 VIL values come back strictly below themselves,
               including every integer in [193,255]. 219 -> 218.99998, so a cell
               whose value is exactly 219 fails `>= 219` and the effective
               threshold at their headline cell becomes 220.
        CUDA : exact for every value except 255, and 255 is not a threshold.

    Their published evaluation runs on CUDA -- `latent_diffusion_model.eval_step`
    calls `eval_metrics.update` on tensors that `data_preprocess` has already
    moved to `self.device` -- so the published numbers are *not* affected, and
    on GPU their scorer and ours agree to 2e-7 on all thirty cells. The CPU
    branch is reachable from the same repo (`test_one_step`, line 257, calls
    `.cpu()` first), so `SEVIRSkillScore` returns a different CSI-219 depending
    on which of its own call sites invoked it. That is a real fragility at
    exactly the extreme threshold, and it is worth one footnote, but it explains
    nothing about the gap and does not touch any published value.

    Kept and carried through the run so the agreement is demonstrated on the
    device that matters rather than assumed. See
    `scripts/verify_scorer_equivalence.py`, which takes the device as an
    argument for this reason.
    """
    return (x / 255.0) / (1. / 255.)


def pool(x, mode):
    """x: (B,T,H,W). Same convention as sevir/pooling_control.py."""
    if mode == "none":
        return x
    k = 4 if mode.endswith("4") else 16
    B, T, H, W = x.shape
    y = x.reshape(B * T, 1, H, W)
    y = (F.avg_pool2d(y, k) if mode.startswith("avg") else F.max_pool2d(y, k))
    return y.reshape(B, T, H // k, W // k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", type=Path,
                    default=P.BASELINES / "cascast_latents" /
                    "earthformer_example" / "test_2h")
    ap.add_argument("--list", type=Path,
                    default=REPO / "datasets/sevir_list/test.txt")
    ap.add_argument("--gt", type=Path, required=True,
                    help="windows h5 from scripts/build_cascast_split.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path(str(P.BASELINES / "cascast")))
    ap.add_argument("--cfg", type=Path,
                    default=REPO / "configs/sevir_used/cascast_diffusion.yaml")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg-weight", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="compute batch. Decoupled from the noise period, "
                         "because 8 frames x 12 lead times does not fit the "
                         "decoder on a 40 GB card")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-period", type=int, default=8,
                    help="their eval runs --batch_size 8 and reseeds the "
                         "generator inside denoise() once per batch, so the "
                         "run holds 8 distinct noise realisations")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(open(args.cfg, "r"))
    ae, cf = build(device, cfg)
    print("loading released checkpoints:")
    n_ae = load_into(ae, args.ckpt_dir / "cascast_casformer.pth",
                     "model.autoencoder_kl", "autoencoder (paired)")
    n_cf = load_into(cf, args.ckpt_dir / "cascast_casformer.pth",
                     "model.casformer", "casformer")
    if any(n_ae) or any(n_cf):
        raise SystemExit("state_dict mismatch; refusing to run")

    sched = DDIMScheduler(**OmegaConf.to_object(
        cfg.model.params.diffusion_kwargs.noise_scheduler.DDPMScheduler))
    sched.set_timesteps(args.steps)

    names = [l.strip() for l in args.list.read_text().splitlines() if l.strip()]
    fg = h5py.File(args.gt, "r")
    G = fg["OUT_vil"]
    stored = [s.decode() if isinstance(s, bytes) else s for s in fg["sample"][:]]
    N = len(names) if not args.limit else min(args.limit, len(names))
    # the GT file was built from the same list, but assert it rather than
    # trusting it: a silent off-by-one here would misalign every pair
    for i in (0, N // 2, N - 1):
        parts = names[i][len("vil-2019-"):-len(".npy")].rsplit(".h5-", 1)
        assert stored[i] == f"{parts[0]}-{parts[1]}", (i, stored[i], names[i])
    print(f"N={N} of {len(names)} samples; gt alignment checked at 3 positions")
    print(f"DDIM steps={args.steps} cfg={args.cfg_weight} "
          f"noise=tiled seed={args.noise_seed} period={args.noise_period}")

    CONV = ("ours", "cascast")
    hits = {c: {p: {t: np.zeros(N, np.int64) for t in THRESHOLDS}
                for p in POOLINGS} for c in CONV}
    fas = {c: {p: {t: np.zeros(N, np.int64) for t in THRESHOLDS}
               for p in POOLINGS} for c in CONV}
    miss = {c: {p: {t: np.zeros(N, np.int64) for t in THRESHOLDS}
                for p in POOLINGS} for c in CONV}

    gen = torch.Generator(device=device)
    bs, period = args.batch_size, args.noise_period
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            z = torch.from_numpy(np.stack([
                np.load(args.latent_dir / names[i]) for i in range(s, e)
            ])).to(device).float()                       # (b,t,4,48,48)
            z = z * SCALE_FACTOR
            b, t = z.shape[0], z.shape[1]

            gen.manual_seed(args.noise_seed)
            npool = torch.randn((period, *z.shape[1:]), generator=gen,
                                device=device)
            lat = torch.stack([npool[i % period] for i in range(s, e)])
            lat = lat * sched.init_noise_sigma

            zz = torch.cat([z, torch.zeros_like(z)])
            for ts in sched.timesteps:
                timestep = torch.ones((b * 2,), device=device) * ts
                inp = sched.scale_model_input(torch.cat([lat] * 2), ts)
                noise = cf(x=inp, timesteps=timestep, cond=zz)
                n_c, n_u = noise.chunk(2)
                noise = n_u + args.cfg_weight * (n_c - n_u)
                lat = sched.step(noise, ts, lat).prev_sample

            dec = ae.net.decode((lat / SCALE_FACTOR).reshape(b * t, *lat.shape[2:]))
            pred = (dec.reshape(b, t, 384, 384).clamp(0, 1) * VIL_SCALE)
            gt = torch.from_numpy(
                G[s:e].astype(np.float32)).to(device).permute(0, 3, 1, 2)

            for conv in CONV:
                # their metric converts units before pooling, so do the same
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
            if (s // bs) % 50 == 0:
                line = []
                for conv in CONV:
                    H = hits[conv]["max16"][219][:e].sum()
                    M = miss[conv]["max16"][219][:e].sum()
                    Fa = fas[conv]["max16"][219][:e].sum()
                    line.append(f"{conv} {H / max(H + M + Fa, 1):.4f}")
                print(f"  {e}/{N}  CSI-219-max16: " + "  ".join(line),
                      flush=True)
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
        "split": "cascast_earthformer_test", "n": N,
        "noise": "tiled", "noise_seed": args.noise_seed,
        "noise_period": period, "batch_size": bs,
        "cfg_weight": args.cfg_weight, "steps": args.steps,
        "latent_source": str(args.latent_dir),
        "table": table,
        "per_sample": {c: {p: {str(t): {"hits": hits[c][p][t].tolist(),
                                        "fas": fas[c][p][t].tolist(),
                                        "miss": miss[c][p][t].tolist()}
                               for t in THRESHOLDS} for p in POOLINGS}
                       for c in CONV},
        "complete": True}))
    print(f"\n{'conv':>8} {'pool':>6} {'thr':>5} {'CSI':>8} {'bias':>8}")
    for r in table:
        print(f"{r['convention']:>8} {r['pooling']:>6} {r['threshold']:>5} "
              f"{r['csi']:8.4f} {r['bias']:8.3f}")
    head = {r["convention"]: r for r in table
            if r["pooling"] == "max16" and r["threshold"] == 219}
    print(f"\nheadline cell, CSI-219 / max-16, published 0.2841:")
    for c in CONV:
        v = head[c]["csi"]
        print(f"  {c:>8} convention: {v:.4f}   {(v - 0.2841) / 0.2841:+.1%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
