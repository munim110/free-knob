"""Third domain: semantic segmentation, where the metric is literally the same one.

Why this closes the domain question
-----------------------------------
Intersection-over-Union, the reporting standard of the segmentation literature,
is algebraically identical to the Critical Success Index:

    IoU = TP / (TP + FP + FN) = h / (h + f + m) = CSI.

So this is not an analogy to the nowcasting metric. It is that metric, computed
by a different community on different data under a different name, at a fixed
operating point, on classes that are often rare. If a free monotone knob moves
CSI at a rare precipitation threshold, it moves IoU on a rare class for exactly
the same reason, and the fact that nobody in segmentation calls it frequency
bias does not make the fixed point disappear.

Setup
-----
Model   torchvision DeepLabV3-ResNet50, the released checkpoint, trained on the
        COCO subset covering the 20 Pascal VOC categories. Not retrained, not
        fine-tuned, not re-implemented.
Data    COCO val2017, ground-truth masks rasterised from the official instance
        annotations for those same 20 categories.
Split   image index parity, matching the SEVIR convention: even = calibration
        (the knob is fitted here and only here), odd = evaluation.
Knob    a shift in logit space, p -> sigmoid(logit(p) + delta). Monotone in p,
        so it reorders no two pixels and adds no information; it only moves the
        operating point. Exactly the free knob of Section 3, in the coordinates
        this domain happens to use.

The rarity axis comes for free: across the 20 categories the positive pixel
fraction spans several orders of magnitude in the same test set, with the model,
the protocol and the split held fixed. That is the controlled sweep of the
rarity section without needing to retrain anything.

Registered prediction
---------------------
Written before the run. The knob's gain should rise as the class gets rarer, and
should be near zero for the common classes. That is the same dose-response,
measured on IoU, with no pooling convention borrowed from anywhere.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

COCO = Path(str(P.COCO))
# torchvision's segmentation checkpoints predict the 21 Pascal VOC classes
# (background + 20). These are their COCO category ids, in VOC label order.
VOC_IN_COCO = [
    ("aeroplane", 5), ("bicycle", 2), ("bird", 16), ("boat", 9),
    ("bottle", 44), ("bus", 6), ("car", 3), ("cat", 17), ("chair", 62),
    ("cow", 21), ("diningtable", 67), ("dog", 18), ("horse", 19),
    ("motorbike", 4), ("person", 1), ("pottedplant", 64), ("sheep", 20),
    ("sofa", 63), ("train", 7), ("tvmonitor", 72),
]
DELTAS = [-4, -3, -2, -1.5, -1, -0.6, -0.3, 0.0, 0.3, 0.6, 1, 1.5, 2, 3, 4]
POOLINGS = [("none", None, 1), ("max8", "max", 8), ("max16", "max", 16)]
# a class needs this many positive pixels in the calibration half before we
# will fit a knob for it at all
MIN_CAL_POS = 20000
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def rasterise(anns, h, w, cat):
    """Binary GT mask for one category, from the official polygon annotations."""
    from pycocotools import mask as cocomask
    segs = [a["segmentation"] for a in anns if a["category_id"] == cat]
    if not segs:
        return np.zeros((h, w), bool)
    out = np.zeros((h, w), bool)
    for s in segs:
        if isinstance(s, list):
            rle = cocomask.frPyObjects(s, h, w)
            rle = cocomask.merge(rle)
        elif isinstance(s["counts"], list):
            rle = cocomask.frPyObjects(s, h, w)
        else:
            rle = s
        out |= cocomask.decode(rle).astype(bool)
    return out


def pool(x, how, k):
    """Non-overlapping reduction, stride = kernel, on a bool or float array."""
    if how is None or k == 1:
        return x
    h, w = x.shape
    nh, nw = h // k, w // k
    if nh == 0 or nw == 0:
        return None
    b = x[:nh * k, :nw * k].reshape(nh, k, nw, k)
    return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))


def counts(pred, gt):
    """IoU and frequency bias from one accumulated contingency table."""
    h = int(np.sum(pred & gt))
    f = int(np.sum(pred & ~gt))
    m = int(np.sum(~pred & gt))
    d, bd = h + f + m, h + m
    return h, f, m, (h / d if d else float("nan")), ((h + f) / bd if bd else
                                                     float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all val2017")
    ap.add_argument("--size", type=int, default=520)
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS / "seg_pooling.json")))
    ap.add_argument("--arch", default="deeplabv3_resnet50")
    a = ap.parse_args()

    from pycocotools.coco import COCO as CocoAPI
    import torchvision.models.segmentation as S

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(S, a.arch)(weights="DEFAULT").to(dev).eval()

    coco = CocoAPI(str(COCO / "annotations/instances_val2017.json"))
    ids = sorted(coco.getImgIds())
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} images, arch={a.arch}, device={dev}")

    # Accumulate the contingency table per (class, pooling, delta, split).
    # Only counts are stored, never probabilities, so memory stays O(1) over
    # 5000 images at 520x520 with 15 knob settings.
    acc = defaultdict(lambda: np.zeros(3, np.int64))
    npos = defaultdict(lambda: np.zeros(2, np.int64))

    for n, iid in enumerate(ids):
        info = coco.loadImgs(iid)[0]
        p = COCO / "val2017" / info["file_name"]
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        W, H = img.size
        split = "cal" if n % 2 == 0 else "eval"

        t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t[None], size=(a.size, a.size), mode="bilinear",
                          align_corners=False)[0]
        t = ((t - MEAN) / STD)[None].to(dev)
        with torch.no_grad():
            logits = model(t)["out"][0]                    # (21, size, size)
        logits = F.interpolate(logits[None], size=(H, W), mode="bilinear",
                               align_corners=False)[0]
        prob = torch.softmax(logits, 0).cpu().numpy()

        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None))
        present = {x["category_id"] for x in anns}
        for vi, (name, cid) in enumerate(VOC_IN_COCO):
            pr = prob[vi + 1]                              # +1 skips background
            gt = rasterise(anns, H, W, cid) if cid in present else None
            if gt is None:
                gt = np.zeros((H, W), bool)
            # logit-space shift of a monotone score: no pixel order changes
            lg = np.log(np.clip(pr, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(pr, 1e-6, 1 - 1e-6)))
            for pl, how, k in POOLINGS:
                g = pool(gt, how, k)
                if g is None:
                    continue
                # max is monotone, so thresholding after pooling is identical to
                # pooling after thresholding: max_i 1[x_i > c] == 1[max_i x_i > c].
                # Pool the logits once instead of once per knob setting. This is
                # exact rather than approximate, and valid only because every
                # pooling here is a max.
                assert how in (None, "max")
                lp = pool(lg, how, k)
                npos[(name, pl, split)] += [int(g.sum()), int(g.size)]
                for d in DELTAS:
                    h, f, m, _, _ = counts(lp + d > 0, g)
                    acc[(name, pl, d, split)] += [h, f, m]
        if (n + 1) % 250 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)

    rows = []
    for (name, pl, split), (pos, tot) in npos.items():
        if split != "eval":
            continue
        R = pos / tot if tot else float("nan")

        def cell(d, sp):
            h, f, m = acc[(name, pl, d, sp)]
            dd, bd = h + f + m, h + m
            return (h / dd if dd else float("nan"),
                    (h + f) / bd if bd else float("nan"))

        base_iou, base_bias = cell(0.0, "eval")
        # The knob is fitted on the calibration half only. A class with almost
        # no positive pixels there cannot support a selection. On a 40-image
        # pilot this picked deltas that drove IoU to zero, so below a floor we
        # decline to fit and report the identity. Declining is the honest
        # outcome rather than a fallback that quietly flatters the method.
        cal_pos = int(npos[(name, pl, "cal")][0])
        cal = {d: cell(d, "cal") for d in DELTAS}
        fittable = cal_pos >= MIN_CAL_POS
        if fittable:
            d_bias = min(DELTAS, key=lambda d: abs(cal[d][1] - 1.0)
                         if np.isfinite(cal[d][1]) else 1e9)
            d_iou = max(DELTAS, key=lambda d: cal[d][0]
                        if np.isfinite(cal[d][0]) else -1)
        else:
            d_bias = d_iou = 0.0
        rows.append({
            "cls": name, "pooling": pl, "R": R, "n_pos": int(pos),
            "iou": base_iou, "bias": base_bias,
            "delta_bias": d_bias, "iou_bias": cell(d_bias, "eval")[0],
            "bias_after": cell(d_bias, "eval")[1],
            "delta_iou": d_iou, "iou_valiou": cell(d_iou, "eval")[0],
            "cal_pos": cal_pos, "fittable": bool(fittable),
            "usable": bool(pos >= 20000 and fittable),
        })
    a.out.write_text(json.dumps(
        {"arch": a.arch, "n_images": len(ids), "size": a.size,
         "deltas": DELTAS, "poolings": [p[0] for p in POOLINGS],
         "table": rows}, indent=1))
    print(f"\nwrote {a.out}")

    for r in sorted([x for x in rows if x["pooling"] == "none"],
                    key=lambda x: -x["R"]):
        if not r["usable"]:
            continue
        print(f"  {r['cls']:14s} R={100*r['R']:6.3f}%  IoU={r['iou']:.4f}  "
              f"bias={r['bias']:6.3f}  d={r['delta_bias']:+.1f}  "
              f"dIoU={r['iou_bias']-r['iou']:+.4f}")


if __name__ == "__main__":
    main()
