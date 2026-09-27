"""ShanghaiTechDataset variant that also returns annotation POINTS.

Bayesian Loss (Ma et al., ICCV 2019) is defined over annotation points rather
than pixels, which is why it is benchmarked here, since
Section 4's scope claim says a loss escapes the impossibility only by
abandoning per-pixel aggregation. Density maps alone cannot express it.

The points must undergo the SAME geometric transforms as the density map, or
the supervision silently decorrelates from the image: the base class applies a
random crop and a random horizontal flip, so points are shifted, filtered to
the crop window, and mirrored to match. The split, the crop size and the
normalisation are inherited unchanged, so a Bayesian-Loss run
sees byte-identical images to every other arm.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.io import loadmat

from dataset import ShanghaiTechDataset


def _load_points(img_path: Path) -> np.ndarray:
    """(N,2) array of (y, x) head coordinates for an image, or (0,2) if absent."""
    gt = img_path.parent.parent / "ground_truth" / f"GT_{img_path.stem}.mat"
    if not gt.exists():
        return np.zeros((0, 2), dtype=np.float32)
    m = loadmat(str(gt))
    try:
        pts = m["image_info"][0, 0][0, 0][0]      # ShanghaiTech layout: (N,2) as (x,y)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return pts[:, [1, 0]].copy()                  # -> (y, x)


class ShanghaiTechPointsDataset(ShanghaiTechDataset):
    """Returns (image, density_map, points) with points transformed to match."""

    def __getitem__(self, idx):
        img_path, dm_path = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        density = np.load(dm_path)
        pts = _load_points(Path(img_path))

        img_t = TF.to_tensor(img)
        dm_t = torch.from_numpy(density).unsqueeze(0).float()
        pts_t = torch.from_numpy(pts)

        if self.augment:
            img_t, dm_t, pts_t = self._augment_with_points(img_t, dm_t, pts_t)
        else:
            _, h0, w0 = img_t.shape
            img_t, dm_t = self._pad_to_divisible(img_t, dm_t, 16)
            # padding is bottom/right only, so point coordinates are unchanged

        img_t = TF.normalize(img_t, self.img_mean, self.img_std)
        return img_t, dm_t, pts_t

    def _augment_with_points(self, img, dm, pts):
        """Mirror of the base class's _augment, applied to points as well."""
        import random
        _, h, w = img.shape
        cs = self.crop_size
        if h < cs or w < cs:
            pad_h, pad_w = max(cs - h, 0), max(cs - w, 0)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            dm = torch.nn.functional.pad(dm, (0, pad_w, 0, pad_h))
            _, h, w = img.shape                     # pads bottom/right: coords unchanged

        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        img = img[:, top:top + cs, left:left + cs]
        dm = dm[:, top:top + cs, left:left + cs]
        if pts.numel():
            pts = pts - torch.tensor([top, left], dtype=pts.dtype)
            keep = ((pts[:, 0] >= 0) & (pts[:, 0] < cs) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < cs))
            pts = pts[keep]

        if random.random() > 0.5:
            img = TF.hflip(img)
            dm = TF.hflip(dm)
            if pts.numel():
                pts = pts.clone()
                pts[:, 1] = (cs - 1) - pts[:, 1]
        return img, dm, pts


def collate_with_points(batch):
    """Points are ragged, so they stay a list; images/maps stack normally."""
    imgs = torch.stack([b[0] for b in batch])
    dms = torch.stack([b[1] for b in batch])
    pts = [b[2] for b in batch]
    return imgs, dms, pts
