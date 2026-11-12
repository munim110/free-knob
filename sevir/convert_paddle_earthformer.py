"""Convert PaddleScience's earthformer_sevir.pdparams to a PyTorch state dict.

The official PyTorch checkpoint (s3://earthformer/pretrained_checkpoints/
earthformer_sevir.pt) has returned 403 since at least 2026-11-09. PaddleScience
hosts the SEVIR weights at
https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_sevir.pdparams.

The file is a pickle. It is read with an Unpickler that admits only numpy's
array reconstructor, ndarray and dtype (the only globals the file references,
checked with pickletools before loading), so no other code can run.

Paddle's nn.Linear stores weight as (in, out); PyTorch's as (out, in). Those
are transposed; every other tensor is copied as is. The script refuses to
write unless every PyTorch key is matched with an identical shape.

    python convert_paddle_earthformer.py --pd earthformer_sevir.pdparams \
        --out earthformer_sevir_from_paddle.pt --dry-run
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy", "ndarray"), ("numpy", "dtype")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refusing global {module}.{name}")
        if module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core"
                                    if hasattr(np, "_core") else "numpy.core")
        return super().find_class(module, name)


def build_model(ef_root):
    sys.path.insert(0, str(ef_root / "src"))
    from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
    cfg = OmegaConf.load(ef_root / "scripts/cuboid_transformer/sevir/earthformer_sevir_v1.yaml")
    m = OmegaConf.to_object(cfg.model)
    n = len(m["enc_depth"])
    for src, dst in (("self_pattern", "enc_attn_patterns"),
                     ("cross_self_pattern", "dec_self_attn_patterns"),
                     ("cross_pattern", "dec_cross_attn_patterns")):
        v = m.pop(src)
        m[dst] = [v] * n if isinstance(v, str) else list(v)
    return CuboidTransformerModel(**m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ef-root", type=Path, default=Path.home() / "baselines" / "ef")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.pd, "rb") as f:
        pd = SafeUnpickler(f, encoding="latin1").load()
    # 'StructuredToParameterName@@' is Paddle's name-mapping metadata and holds
    # no tensor; one released key carries a trailing newline.
    pd = {k.strip(): np.asarray(v) for k, v in pd.items()
          if k != "StructuredToParameterName@@"}
    model = build_model(args.ef_root)
    linear = {n + ".weight" for n, mod in model.named_modules()
              if isinstance(mod, torch.nn.Linear)}
    ref = model.state_dict()

    # Paddle names BatchNorm statistics differently; nothing else should differ.
    rename = {"._mean": ".running_mean", "._variance": ".running_var"}

    def tkey(k):
        for a, b in rename.items():
            if k.endswith(a):
                return k[: -len(a)] + b
        return k

    out, bad, extra = {}, [], []
    for k, v in pd.items():
        tk = tkey(k)
        if tk not in ref:
            extra.append(k)
            continue
        t = torch.from_numpy(np.ascontiguousarray(v.T if tk in linear and v.ndim == 2 else v))
        if tuple(t.shape) != tuple(ref[tk].shape):
            bad.append((tk, tuple(v.shape), tuple(ref[tk].shape)))
            continue
        out[tk] = t.to(ref[tk].dtype)
    missing = [k for k in ref if k not in out]
    print(f"paddle keys {len(pd)}  torch keys {len(ref)}  matched {len(out)}")
    print(f"unmatched paddle {len(extra)}  shape mismatches {len(bad)}  missing torch {len(missing)}")
    for k in extra[:10]:
        print("  paddle-only", k, pd[k].shape)
    for b in bad[:10]:
        print("  shape", b)
    for k in missing[:10]:
        print("  torch-only", k, tuple(ref[k].shape))
    if args.dry_run:
        return
    if extra or bad or missing:
        sys.exit("refusing to write: the mapping is not one-to-one")
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
