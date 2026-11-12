"""DiffCast (CVPR 2024) on SEVIR-128: the cascade forecast and its own PhyDNet backbone.

Uses the released checkpoint (resources/diffcast_phydnet_sevir128.pt from the
DiffCast README) and the DiffCast repository's model and data code, with the
published test protocol: SEVIR VIL from 2019-10-01, resized to 128x128,
5 input frames, 20 output frames, windows of 25 frames at stride 5 (five per
event), 250 DDIM steps. One call to GaussianDiffusion.sample returns both the
DiffCast forecast and the backbone's deterministic forecast, so each window
gives the pair.

Frames are stored as uint8 with DiffCast's own float2int (clip to [0,1],
multiply by 255, truncate), so scoring them reproduces the published metric.

torchvision note: DiffCast ran with torchvision 0.13, whose tensor Resize does
not antialias by default; current torchvision does. antialias=False is pinned
to match the published inputs.
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision import transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "baselines/DiffCast"))
    ap.add_argument("--ckpt", default=str(Path.home() / "baselines/diffcast_weights/diffcast_phydnet_sevir128.pt"))
    ap.add_argument("--data", default=str(Path.home() / "data/sevir_full"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--event-stride", type=int, default=1,
                    help="keep every k-th event (all five windows of each kept event)")
    ap.add_argument("--event-offset", type=int, default=0,
                    help="first event index; offsets 0..k-1 with stride k partition the test split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", action="store_true", help="use EMA weights (the published eval uses the online model)")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    import datetime
    from datasets.dataset_sevir import SEVIRTorchDataset
    from models.phydnet import get_model as get_phydnet
    from diffcast import get_model as get_diffcast

    dev = torch.device("cuda")
    backbone = get_phydnet(in_shape=(1, 128, 128), T_in=5, T_out=20, device=dev)
    model = get_diffcast(img_channels=1, dim=64, dim_mults=(1, 2, 4, 8), T_in=5, T_out=20,
                         sampling_timesteps=250)
    model.load_backbone(backbone)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print("checkpoint keys:", list(sd)[:6] if isinstance(sd, dict) else type(sd))
    state = sd["ema"] if args.ema else sd.get("model", sd)
    if args.ema:  # EMA wrapper keys: ema_model.<...>
        state = {k[len("ema_model."):]: v for k, v in state.items() if k.startswith("ema_model.")}
    print(model.load_state_dict(state, strict=True))
    model.to(dev).eval()

    ds = SEVIRTorchDataset(dataset_dir=args.data, split_mode="uneven", shuffle=False, img_size=128,
                           seq_len=25, stride=5, sample_mode="sequent", batch_size=5, num_shard=1, rank=0,
                           start_date=datetime.datetime(2019, 10, 1), end_date=None,
                           output_type=np.float32, preprocess=True, rescale_method="01", verbose=False)
    ds.transform = transforms.Compose([transforms.Resize((128, 128), antialias=False)])
    dl = ds.sevir_dataloader
    n_event = dl.total_num_event
    assert dl.num_seq_per_event == 5
    events = list(range(args.event_offset, n_event, args.event_stride))
    if args.max_batches:
        events = events[: max(1, args.max_batches * args.batch_size // 5)]
    ids = [str(dl._samples.iloc[e].name) for e in events]
    print(f"test events: {n_event}; using {len(events)} events = {5 * len(events)} windows", flush=True)

    torch.manual_seed(args.seed)
    N = 5 * len(events)
    with h5py.File(args.out, "w") as f:
        kw = dict(dtype="uint8", chunks=(5, 20, 128, 128), compression="lzf")
        dg = f.create_dataset("gt", (N, 20, 128, 128), **kw)
        dp = f.create_dataset("diffcast", (N, 20, 128, 128), **kw)
        db = f.create_dataset("backbone", (N, 20, 128, 128), **kw)
        dl5 = f.create_dataset("last_input", (N, 128, 128), dtype="uint8")  # persistence
        f.create_dataset("event_id", data=np.repeat(np.array(ids, dtype="S"), 5))
        f.create_dataset("window", data=np.tile(np.arange(5), len(events)))
        f.attrs.update(ckpt=args.ckpt, seed=args.seed, ema=args.ema, ddim_steps=250,
                       event_stride=args.event_stride, event_offset=args.event_offset, n_test_events=n_event)

        def to_u8(x):  # DiffCast's float2int
            return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint16).astype(np.uint8)

        per = max(1, args.batch_size // 5)
        t0, done = time.time(), 0
        for s in range(0, len(events), per):
            evs = events[s:s + per]
            seqs = torch.cat([ds[e] for e in evs], 0)  # ds[e]: one event's 5 windows, (5, 25, 1, 128, 128)
            x = seqs.to(dev)
            with torch.no_grad():
                pred, mu, _ = model.sample(frames_in=x[:, :5], T_out=20)
            sl = slice(done, done + x.shape[0])
            dg[sl] = to_u8(seqs[:, 5:, 0].numpy())
            dp[sl] = to_u8(pred[:, :, 0].float().cpu().numpy())
            db[sl] = to_u8(mu[:, :, 0].float().cpu().numpy())
            dl5[sl] = to_u8(seqs[:, 4, 0].numpy())
            done += x.shape[0]
            el = time.time() - t0
            print(f"  {done}/{N} windows  {el:.0f}s  eta {el / done * (N - done) / 60:.0f} min", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
