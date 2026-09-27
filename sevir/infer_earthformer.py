"""Run the released Earthformer SEVIR checkpoint over the official preprocessed
nowcast test split and cache its predictions in VIL units.

The model is built by the repo's own `config_cuboid_transformer` recipe from
`earthformer_sevir_v1.yaml` and loaded from the released `earthformer_sevir.pt`,
so no part of it is a reimplementation. Preprocessing follows the repo's own
rescale='01' convention for VIL (x/255 in, x*255 out).

Predictions are cached so the calibration control can be applied post hoc,
without re-running the network once per knob setting.
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402

EF_ROOT = P.BASELINES / "ef"
sys.path.insert(0, str(EF_ROOT / "src"))

from earthformer.cuboid_transformer.cuboid_transformer import (  # noqa: E402
    CuboidTransformerModel)

VIL_SCALE = 255.0  # repo's rescale='01' for vil


def config_cuboid_transformer(cfg):
    """Verbatim from tests/unittests/test_pretrained_checkpoints.py."""
    model_cfg = OmegaConf.to_object(cfg.model)
    num_blocks = len(model_cfg["enc_depth"])
    for src, dst in (("self_pattern", "enc_attn_patterns"),
                     ("cross_self_pattern", "dec_self_attn_patterns"),
                     ("cross_pattern", "dec_cross_attn_patterns")):
        v = model_cfg.pop(src)
        model_cfg[dst] = [v] * num_blocks if isinstance(v, str) else list(v)
    return CuboidTransformerModel(**model_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path,
                    default=P.SEVIR_H5)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(str(P.BASELINES / "earthformer_sevir.pt")))
    ap.add_argument("--cfg", type=Path,
                    default=EF_ROOT / "scripts/cuboid_transformer/sevir/"
                                      "earthformer_sevir_v1.yaml")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/"
                                 "sevir_pred_earthformer.h5"))
    ap.add_argument("--ckpt-path", type=str, default=None,
                    help="dotted path into a nested checkpoint, e.g. "
                         "'model.EarthFormer_xy' for CasCast's release")
    ap.add_argument("--strip-prefix", type=str, default=None,
                    help="prefix to strip from state_dict keys, e.g. 'net.'")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all events")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(open(args.cfg, "r"))
    model = config_cuboid_transformer(cfg).to(device)
    sd = torch.load(args.ckpt, map_location="cpu",
                    weights_only=args.ckpt_path is None)
    if args.ckpt_path:
        for part in args.ckpt_path.split("."):
            sd = sd[part]
    if args.strip_prefix:
        p = args.strip_prefix
        sd = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:5]:
            print("   missing", k)
        for k in list(unexpected)[:5]:
            print("   unexpected", k)
    else:
        print("state_dict loaded strictly clean")
    model.eval()

    f = h5py.File(args.h5, "r")
    IN, OUT = f["IN_vil"], f["OUT_vil"]
    N = IN.shape[0] if not args.limit else min(args.limit, IN.shape[0])
    print(f"events={N} in={IN.shape} out={OUT.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    g = h5py.File(args.out, "w")
    dset = g.create_dataset("pred_vil", shape=(N, 384, 384, 12),
                            dtype="float32", chunks=(1, 384, 384, 12),
                            compression="lzf")

    bs = args.batch_size
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            # h5 layout (N,H,W,T) -> model layout NTHWC
            x = IN[s:e].astype(np.float32) / VIL_SCALE
            x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(-1).to(device)
            y = model(x)                       # (B,12,384,384,1) in [0,1]
            y = y.squeeze(-1).permute(0, 2, 3, 1)   # -> (B,384,384,12)
            y = (y.clamp(0, 1) * VIL_SCALE).cpu().numpy().astype(np.float32)
            dset[s:e] = y
            if (s // bs) % 10 == 0:
                print(f"  {e}/{N}", flush=True)
    g.attrs["model"] = "earthformer_sevir"
    g.attrs["scale"] = VIL_SCALE
    g.close()
    f.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
