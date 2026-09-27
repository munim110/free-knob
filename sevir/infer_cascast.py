"""Run CasCast's full cascade (deterministic backbone -> VAE latent -> CasFormer
diffusion -> decode) on the SEVIR nowcast test split.

Built from the authors' released checkpoints and their own eval code path
(`models/latent_diffusion_model_eval.py`) rather than from a reimplementation:

    z_coarse = AE.encode(coarse).sample() * scale_factor
    latents  = randn * init_noise_sigma
    for t in DDIM(20).timesteps:
        latents = step(CasFormer(latents, t, cond=z_coarse), t, latents)
    out      = AE.decode(latents / scale_factor)

scale_factor = 0.6786020398139954 is the value hardcoded in their
`test_final` for the SEVIRSkillScore evaluation.

Classifier-free guidance: their released `scripts/eval_diffusion_infer.sh` runs
with `--cfg_weight 2 --ens_member 1`, which takes the `cfg != 1` branch of
`denoise()`, which uses a doubled batch with a zeroed condition, combined as
`uncond + cfg * (cond - uncond)`. Running the plain conditional path instead
undershoots their reported CSI-219-POOL16 by ~13%, so `--cfg 2` is the setting
that reproduces the paper and is the default here. `--cfg 1` reproduces the
unguided sampler for reference.

The coarse prediction is taken from a cached HDF5 produced by running their own
deterministic checkpoint, so the only thing this script adds is the cascade.

Sampling noise. The published run used `--noise tiled --noise-seed 0`, which
reseeds the generator to a fixed value before every batch and draws one tensor of
shape (batch, t, 4, 48, 48). Every full batch therefore receives the *same* noise
tensor, so across the test set the latent noise repeats with period
`--batch-size`: with the default of 4, events i and i+4 are sampled from
identical noise. It means the run holds four distinct noise realisations rather
than N independent ones, so
a pooled score over events carries an effective sample size of `--batch-size` in
the sampling dimension and an event-level bootstrap cannot see that component at
all. `--noise independent` draws one realisation per event, keyed by the global
event index so it does not depend on the batch size; varying `--noise-seed` over
that setting is what puts sampling stochasticity into the interval. `tiled`
remains the default so the numbers in the paper reproduce unchanged.

This script is NOT bit-reproducible run to run, and an earlier version of this
docstring wrongly claimed it was. `--noise-seed` fixes the diffusion noise, but
the conditioning latent is `encode(coarse).sample()`, a draw from the VAE
posterior taken through `randn_tensor(generator=None)` -- the unseeded global
RNG. Two runs with identical arguments differ by up to about 5 VIL units per
pixel for that reason. CasCast has the same term and freezes it by accident, by
caching the latents once and evaluating from the cache ever after, so their
published number is conditioned on one particular unrecorded draw of encoder
noise. The seed families measured in `docs/preregistration/` therefore carry
diffusion noise and encoder noise together, which is the right total for an
interval on a re-run but is not a decomposition of the two.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

REPO = Path(str(P.BASELINES / "cascast_repo"))
sys.path.insert(0, str(REPO))

from networks.autoencoder_kl import autoencoder_kl        # noqa: E402
from networks.casformer import CasFormer                  # noqa: E402
from src.diffusers import DDIMScheduler                   # noqa: E402


SCALE_FACTOR = 0.6786020398139954
VIL_SCALE = 255.0


def build(device, cfg):
    p = cfg.model.params.sub_model
    ae = autoencoder_kl(OmegaConf.to_object(p.autoencoder_kl)).to(device).eval()
    cf = CasFormer(arch=p.casformer.arch,
                   config=OmegaConf.to_object(p.casformer.config)).to(device).eval()
    return ae, cf


def load_into(module, path, key_path, strip, label):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for part in key_path.split("."):
        sd = sd[part]
    if strip:
        sd = {k[len(strip):]: v for k, v in sd.items() if k.startswith(strip)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    print(f"  {label}: {len(sd)} tensors, missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    if missing:
        print(f"    e.g. missing {list(missing)[:3]}")
    if unexpected:
        print(f"    e.g. unexpected {list(unexpected)[:3]}")
    return len(missing), len(unexpected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=Path,
                    default=Path(str(P.RESULTS) + "/"
                                 "sevir_pred_cascast_det.h5"))
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/"
                                 "sevir_pred_cascast_cascade.h5"))
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path(str(P.BASELINES / "cascast")))
    ap.add_argument("--cfg", type=Path,
                    default=REPO / "configs/sevir_used/cascast_diffusion.yaml")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg-weight", type=float, default=2.0,
                    help="classifier-free guidance weight; 2 matches their "
                         "released eval script, 1 disables guidance")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--noise", choices=("tiled", "independent"), default="tiled",
                    help="tiled reproduces the published run: one noise tensor "
                         "reused by every batch, so noise repeats with period "
                         "--batch-size. independent draws one realisation per "
                         "event, keyed by global event index")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-period", type=int, default=0,
                    help="under --noise tiled, the number of distinct noise "
                         "realisations the run holds. 0 means use --batch-size, "
                         "which is what the released eval does (its denoise() "
                         "reseeds per batch). Setting it decouples the noise "
                         "period from the compute batch, so the noise scheme of "
                         "a batch-size-8 run can be reproduced on a GPU that "
                         "only fits 4. Event i receives slice i %% period, which "
                         "for period == batch-size is bit-identical to the "
                         "batch-reseeding path.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(open(args.cfg, "r"))
    ae, cf = build(device, cfg)

    print("loading released checkpoints:")
    # Both state dicts already carry the attribute prefix the modules expect
    # (autoencoder_kl.net.*, CasFormer.model.*), so nothing is stripped. The
    # autoencoder is taken from the CasFormer checkpoint because that is the
    # one the diffusion model was trained against.
    n_ae = load_into(ae, args.ckpt_dir / "cascast_casformer.pth",
                     "model.autoencoder_kl", None, "autoencoder (paired)")
    n_cf = load_into(cf, args.ckpt_dir / "cascast_casformer.pth",
                     "model.casformer", None, "casformer")
    if n_ae[0] or n_ae[1] or n_cf[0] or n_cf[1]:
        raise SystemExit("state_dict mismatch. Refusing to run a model that "
                         "did not load cleanly, since its output would look "
                         "plausible and be wrong")

    sched = DDIMScheduler(
        **OmegaConf.to_object(
            cfg.model.params.diffusion_kwargs.noise_scheduler.DDPMScheduler))
    sched.set_timesteps(args.steps)
    print(f"DDIM steps={args.steps} scale_factor={SCALE_FACTOR} cfg={args.cfg_weight}")

    fc = h5py.File(args.coarse, "r")
    C = fc["pred_vil"]
    N = C.shape[0] if not args.limit else min(args.limit, C.shape[0])
    g = h5py.File(args.out, "w")
    dset = g.create_dataset("pred_vil", shape=(N, 384, 384, 12), dtype="float32",
                            chunks=(1, 384, 384, 12), compression="lzf")

    gen = torch.Generator(device=device)
    bs = args.batch_size
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            c = torch.from_numpy(C[s:e].astype(np.float32) / VIL_SCALE).to(device)
            c = c.permute(0, 3, 1, 2).unsqueeze(2)          # (b,t,1,H,W)
            b, t = c.shape[0], c.shape[1]
            flat = c.reshape(b * t, 1, 384, 384)
            z = ae.net.encode(flat).sample() * SCALE_FACTOR
            z = z.reshape(b, t, *z.shape[1:])               # (b,t,4,48,48)

            if args.noise == "tiled":
                period = args.noise_period or bs
                gen.manual_seed(args.noise_seed)
                pool = torch.randn((period, *z.shape[1:]), generator=gen,
                                   device=device)
                lat = torch.stack([pool[i % period] for i in range(s, e)])
            else:
                # Key each event's noise to its global index rather than to its
                # position in a batch, so the realisation an event receives is
                # invariant to --batch-size and reproducible on its own.
                draws = []
                for i in range(s, e):
                    gen.manual_seed(args.noise_seed * 1_000_003 + i)
                    draws.append(
                        torch.randn(z.shape[1:], generator=gen, device=device))
                lat = torch.stack(draws)
            lat = lat * sched.init_noise_sigma
            if args.cfg_weight == 1:
                for ts in sched.timesteps:
                    timestep = torch.ones((b,), device=device) * ts
                    noise = cf(x=lat, timesteps=timestep, cond=z)
                    lat = sched.step(noise, ts, lat).prev_sample
            else:
                # their denoise() cfg branch, verbatim in structure: the
                # condition is duplicated with a zeroed second half, so
                # chunk(2) gives (conditional, unconditional) in that order
                zz = torch.cat([z, torch.zeros_like(z)])
                for ts in sched.timesteps:
                    timestep = torch.ones((b * 2,), device=device) * ts
                    inp = sched.scale_model_input(torch.cat([lat] * 2), ts)
                    noise = cf(x=inp, timesteps=timestep, cond=zz)
                    n_c, n_u = noise.chunk(2)
                    noise = n_u + args.cfg_weight * (n_c - n_u)
                    lat = sched.step(noise, ts, lat).prev_sample

            dec = ae.net.decode(
                (lat / SCALE_FACTOR).reshape(b * t, *lat.shape[2:]))
            dec = dec.reshape(b, t, 384, 384).permute(0, 2, 3, 1)
            dset[s:e] = (dec.clamp(0, 1) * VIL_SCALE).cpu().numpy().astype(
                np.float32)
            if (s // bs) % 10 == 0:
                print(f"  {e}/{N}", flush=True)

    g.attrs["model"] = "cascast_cascade"
    g.attrs["ddim_steps"] = args.steps
    g.attrs["cfg_weight"] = args.cfg_weight
    g.attrs["noise"] = args.noise
    g.attrs["noise_seed"] = args.noise_seed
    g.attrs["batch_size"] = args.batch_size
    g.attrs["noise_period"] = args.noise_period or args.batch_size
    g.close(); fc.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
