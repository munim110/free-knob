"""
ShanghaiTech dataset for crowd density estimation.
Loads RGB images and pre-generated density maps.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
import random


# Deterministic validation split: 30 images held out from train_data
# Selected by sorted image name modulo: every 10th image starting from index 5
# (i.e., IMG_5, IMG_15, IMG_25, ..., IMG_295) → 30 val images, 270 train images.
# This avoids any contamination with test_data and is reproducible.
def get_train_val_split(data_dir: Path, part: str, val_indices=None):
    """
    Returns (train_image_nums, val_image_nums) for a deterministic split.
    """
    img_dir = data_dir / part / "train_data" / "images"
    all_imgs = sorted(img_dir.glob("*.jpg"))
    nums = sorted([int(p.stem.split("_")[-1]) for p in all_imgs])

    if val_indices is None:
        # Hold out every 10th image (numbered 5, 15, 25, ...)
        val_set = set(nums[5::10])  # every 10th starting from index 5
    else:
        val_set = set(val_indices)

    train_nums = [n for n in nums if n not in val_set]
    val_nums = [n for n in nums if n in val_set]
    return train_nums, val_nums


class ShanghaiTechDataset(Dataset):
    """
    Dataset for ShanghaiTech crowd density estimation.

    Returns (image, density_map) pairs. Training applies random crops
    and horizontal flips. Test returns full images.
    """

    def __init__(self, data_dir: Path, part: str, split: str,
                 crop_size: int = 256, augment: bool = True,
                 use_train_val_split: bool = False):
        """
        Args:
            data_dir: Root of ShanghaiTech dataset
            part: 'part_A_final' or 'part_B_final'
            split: 'train' (270 imgs), 'val' (30 imgs), 'test' (182 imgs).
                   Also accepts 'train_data' / 'test_data' for legacy use,
                   which loads the FULL train/test set without val split.
            crop_size: Size of random crop for training
            augment: Whether to apply augmentation (train only)
            use_train_val_split: If True, splits train_data into train+val
        """
        self.crop_size = crop_size

        # Determine which physical directory to read from and which images to use
        if split in ('train_data', 'test_data'):
            # Legacy: full train or test, no val split
            phys_split = split
            self.augment = augment and (split == "train_data")
            allowed_nums = None
        elif split in ('train', 'val'):
            # New: train_data physical, but split into train/val
            phys_split = 'train_data'
            self.augment = augment and (split == 'train')
            train_nums, val_nums = get_train_val_split(data_dir, part)
            allowed_nums = set(train_nums) if split == 'train' else set(val_nums)
        elif split == 'test':
            phys_split = 'test_data'
            self.augment = False
            allowed_nums = None
        else:
            raise ValueError(f"Unknown split: {split}")

        img_dir = data_dir / part / phys_split / "images"
        dm_dir = data_dir / part / phys_split / "density_maps"

        self.samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            num = int(img_path.stem.split("_")[-1])
            if allowed_nums is not None and num not in allowed_nums:
                continue
            dm_path = dm_dir / f"density_IMG_{num}.npy"
            if dm_path.exists():
                self.samples.append((img_path, dm_path))

        if not self.samples:
            raise FileNotFoundError(f"No samples found in {img_dir} for split={split}")

        # Compute normalization stats from training set
        self.img_mean = [0.485, 0.456, 0.406]  # ImageNet
        self.img_std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, dm_path = self.samples[idx]

        # Load image and density map
        img = Image.open(img_path).convert("RGB")
        density = np.load(dm_path)

        # Convert to tensors
        img_tensor = TF.to_tensor(img)  # (3, H, W), [0, 1]
        dm_tensor = torch.from_numpy(density).unsqueeze(0).float()  # (1, H, W)

        if self.augment:
            img_tensor, dm_tensor = self._augment(img_tensor, dm_tensor)
        else:
            # For test: ensure dimensions are divisible by 16 (for encoder)
            img_tensor, dm_tensor = self._pad_to_divisible(img_tensor, dm_tensor, 16)

        # Normalize image
        img_tensor = TF.normalize(img_tensor, self.img_mean, self.img_std)

        return img_tensor, dm_tensor

    def _augment(self, img, dm):
        """Random crop + horizontal flip."""
        _, h, w = img.shape
        cs = self.crop_size

        # Ensure image is at least crop_size
        if h < cs or w < cs:
            pad_h = max(cs - h, 0)
            pad_w = max(cs - w, 0)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            dm = torch.nn.functional.pad(dm, (0, pad_w, 0, pad_h))
            _, h, w = img.shape

        # Random crop
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        img = img[:, top:top+cs, left:left+cs]
        dm = dm[:, top:top+cs, left:left+cs]

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            dm = TF.hflip(dm)

        return img, dm

    def _pad_to_divisible(self, img, dm, divisor):
        """Pad to make dimensions divisible by divisor."""
        _, h, w = img.shape
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            dm = torch.nn.functional.pad(dm, (0, pad_w, 0, pad_h))
        return img, dm

    def get_image_name(self, idx):
        return self.samples[idx][0].stem
