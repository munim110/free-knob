"""PreDiff (NeurIPS 2023) on SEVIR-LR: forecasts on the published test split.

Uses the PreDiff repository's own module, config (prediff_sevirlr_v1.yaml) and
data module, and the authors' released weights (VAE, latent Earthformer-UNet;
the knowledge-alignment network is built by the module but not used, because
the paper reports no CSI for PreDiff-KA). Test protocol as in the repository:
SEVIR-LR from 2019-06-01, 7 context frames, 6 target frames, windows of 13
frames at stride 6, the full 1000-step ancestral sampler, no EMA swap (the
repository's test_step samples with the loaded weights).

Frames are stored as uint8 floor(clip(x, 0, 1) * 255). PreDiff's evaluator
compares x * 255 with integer thresholds, which this storage preserves exactly.
Several samples per context can be stored (--samples); the paper scores CSI on
the mean of an unstated number of samples, the released config on one sample.

Prerequisite links (made by the caller):
  PreDiff/datasets/sevirlr -> SEVIR-LR (CATALOG.csv, data/vil/2017..2019)
  PreDiff/pretrained/{vae,earthformerunet,alignment}/<released .pt>
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "baselines/PreDiff"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1, help="samples per context")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--window-stride", type=int, default=1, help="keep every k-th test window")
    ap.add_argument("--window-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    script_dir = os.path.join(args.repo, "scripts/prediff/sevirlr")
    sys.path.insert(0, script_dir)
    from omegaconf import OmegaConf
    import train_sevirlr_prediff
    from train_sevirlr_prediff import PreDiffSEVIRPLModule
    # FVD is not needed here and would download I3D weights; stub the metric out
    train_sevirlr_prediff.FrechetVideoDistance = lambda *a, **k: torch.nn.Identity()
    from prediff.utils.path import default_pretrained_earthformerunet_dir
    from prediff.utils.download import pretrained_sevirlr_earthformerunet_name

    cfg = os.path.join(script_dir, "prediff_sevirlr_v1.yaml")
    oc = OmegaConf.load(open(cfg))
    torch.set_float32_matmul_precision(oc.optim.float32_matmul_precision)
    dataset_cfg = OmegaConf.to_object(oc.dataset)
    dm = PreDiffSEVIRPLModule.get_sevir_datamodule(dataset_cfg=dataset_cfg, micro_batch_size=1, num_workers=4)
    dm.prepare_data()
    dm.setup(stage="test")
    test = dm.sevir_test
    n_all = len(test)
    idx = list(range(args.window_offset, n_all, args.window_stride))
    if args.max_batches:
        idx = idx[: args.max_batches * args.batch_size]
    print(f"test windows: {n_all}; using {len(idx)}", flush=True)

    pl_module = PreDiffSEVIRPLModule(total_num_steps=1, oc_file=cfg, save_dir=str(Path(args.out).parent / "prediff_tmp"))
    sd = torch.load(os.path.join(default_pretrained_earthformerunet_dir, pretrained_sevirlr_earthformerunet_name),
                    map_location="cpu", weights_only=True)
    print(pl_module.torch_nn_module.load_state_dict(sd, strict=True))
    dev = torch.device("cuda")
    pl_module = pl_module.to(dev).eval()

    def to_u8(x):
        return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint16).astype(np.uint8)

    torch.manual_seed(args.seed)
    N, K = len(idx), args.samples
    with h5py.File(args.out, "w") as f:
        kw = dict(dtype="uint8", compression="lzf")
        dg = f.create_dataset("gt", (N, 6, 128, 128), chunks=(1, 6, 128, 128), **kw)
        dp = f.create_dataset("prediff", (N, K, 6, 128, 128), chunks=(1, 1, 6, 128, 128), **kw)
        dl = f.create_dataset("last_input", (N, 128, 128), dtype="uint8")
        f.create_dataset("window_index", data=np.array(idx))
        f.attrs.update(seed=args.seed, samples=K, n_test_windows=n_all, sampler_steps=int(pl_module.num_timesteps),
                       window_stride=args.window_stride, window_offset=args.window_offset)
        t0 = time.time()
        for s in range(0, N, args.batch_size):
            ii = idx[s:s + args.batch_size]
            batch = torch.stack([torch.as_tensor(test[i]) for i in ii]).to(dev)  # (B, 13, 128, 128, 1), NTHWC
            target, cond, context = pl_module.get_input(batch, return_verbose=True)
            preds = []
            with torch.no_grad():
                for _ in range(K):
                    preds.append(pl_module.sample(cond=cond, batch_size=len(ii), use_alignment=False,
                                                  verbose=False).float().cpu().numpy()[..., 0])
            sl = slice(s, s + len(ii))
            dg[sl] = to_u8(target.float().cpu().numpy()[..., 0])
            dp[sl] = to_u8(np.stack(preds, 1))
            dl[sl] = to_u8(context[:, -1, ..., 0].float().cpu().numpy())
            done = s + len(ii)
            el = time.time() - t0
            print(f"  {done}/{N} windows  {el:.0f}s  eta {el / done * (N - done) / 60:.0f} min", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
