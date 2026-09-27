"""Reproduction check for the segmentation pipeline.

The paper's per-class IoU thresholds the class probability, because a threshold
is what gives us an operating point to move. Standard mIoU instead takes the
argmax over classes. The two differ, since p_c > 0.5 implies argmax = c but
the converse fails, so before reading anything into the numbers we confirm that the same
loading, resizing and mask-rasterisation path reproduces the published argmax
mIoU for this checkpoint. If it does, the pipeline is sound and the difference
between the two columns is the operating point, which is the paper's subject.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from pooling_control import (COCO, VOC_IN_COCO, MEAN, STD, rasterise)

import argparse

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-n", "--images", type=int, default=500,
                 dest="n_images",
                 help="how many val2017 images to score (default 500; the paper "
                      "reports 1000)")
N = _ap.parse_args().n_images
from pycocotools.coco import COCO as CocoAPI
import torchvision.models.segmentation as S

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = S.deeplabv3_resnet50(weights="DEFAULT").to(dev).eval()
coco = CocoAPI(str(COCO / "annotations/instances_val2017.json"))
ids = sorted(coco.getImgIds())[:N]

inter = np.zeros(21, np.int64)
union = np.zeros(21, np.int64)
for n, iid in enumerate(ids):
    info = coco.loadImgs(iid)[0]
    img = Image.open(COCO / "val2017" / info["file_name"]).convert("RGB")
    W, H = img.size
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    t = F.interpolate(t[None], size=(520, 520), mode="bilinear",
                      align_corners=False)[0]
    with torch.no_grad():
        lg = model(((t - MEAN) / STD)[None].to(dev))["out"][0]
    lg = F.interpolate(lg[None], size=(H, W), mode="bilinear",
                       align_corners=False)[0]
    pred = lg.argmax(0).cpu().numpy()

    anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None))
    present = {a["category_id"] for a in anns}
    gt = np.zeros((H, W), np.int64)          # 0 = background
    for vi, (name, cid) in enumerate(VOC_IN_COCO):
        if cid in present:
            gt[rasterise(anns, H, W, cid)] = vi + 1
    for c in range(21):
        p, g = pred == c, gt == c
        inter[c] += int((p & g).sum())
        union[c] += int((p | g).sum())
iou = inter / np.maximum(union, 1)
import json

PUBLISHED = 0.664
json.dump({"n_images": N, "miou": float(iou.mean()), "published": PUBLISHED,
           "rel_dev": float((iou.mean() - PUBLISHED) / PUBLISHED),
           "per_class": {n: float(iou[i]) for i, (n, _) in
                         enumerate([("background", 0)] + VOC_IN_COCO)},
           # two known protocol differences, both of which depress our number:
           # we score every val2017 image (torchvision's subset keeps only those
           # containing a VOC category, so we take extra false alarms), and we
           # keep iscrowd regions in the target instead of ignoring them.
           "known_differences": ["all val2017 images rather than the VOC subset",
                                 "iscrowd regions kept in the target"]},
          open(str(P.RESULTS / "seg_repro.json"), "w"), indent=1)
print(f"\n{N} images, argmax mIoU over 21 classes = {iou.mean():.4f}")
print("torchvision reports 0.664 for deeplabv3_resnet50 on this subset")
for c, (name, _) in enumerate([("background", 0)] + VOC_IN_COCO):
    print(f"  {name:14s} {iou[c]:.4f}")
