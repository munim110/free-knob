"""Radar domain: what the matched control does to a dual-vs-single comparison.

The original protocol gave the single decoder the post-hoc knob family
{gain, qmap} and the dual decoder only its architectural blend weight beta.
`posthoc_radar_dual_decoder_matched_s42.json` re-runs the dual decoder with the same
knob family added, so both sides can be scored under one rule.

Operating point: the arm whose frequency bias is closest to 1, chosen without
reference to the arm's own CSI. There is no separate validation split in these
cached files, so this is a same-split selection and is reported as such. It is
applied identically to both sides, which is the property the comparison needs.
Only one dual-decoder seed exists at matched settings, so this is a single-seed
result and is presented as corroboration of the SEVIR analysis, not as
independent evidence of an effect size.
"""
import argparse
import json
from pathlib import Path

BETA = "beta="
THRESH = ["0.1mm", "0.5mm", "1mm", "2.5mm", "5mm"]


def best(arms, thr, families):
    """Arm with |bias - 1| minimal among the named families; returns (csi, tag)."""
    cand = []
    for tag, a in arms.items():
        if not any(tag.startswith(f) for f in families):
            continue
        cell = a["per_threshold"].get(thr)
        if not cell:
            continue
        cand.append((abs(cell["bias"] - 1.0), cell["csi"], tag))
    if not cand:
        return None, None
    _, csi, tag = min(cand)
    return csi, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual_decoder", type=Path,
                    default=Path("results/posthoc_radar_dual_decoder_matched_s42.json"))
    ap.add_argument("--single", nargs="+", type=Path,
                    default=[Path("results/posthoc_radar_plain_mse_s42.json"),
                             Path("results/posthoc_radar_plain_mse_s43.json"),
                             Path("results/posthoc_radar_plain_mse_s44.json")])
    ap.add_argument("--out", type=Path,
                    default=Path("results/radar_matched_report.md"))
    args = ap.parse_args()

    D = json.loads(args.dual_decoder.read_text())["arms"]
    S = [json.loads(p.read_text())["arms"] for p in args.single]

    L = ["# Radar (DWD RY / RainNet): matched vs asymmetric control\n",
         "`dual (beta only)` is the original protocol: the dual decoder may "
         "move only its architectural blend weight. `dual (matched)` gives it "
         "the same {gain, qmap} family the single decoder always had. The "
         "single-decoder column is the mean over three seeds, each selected by "
         "the same rule.\n",
         "| threshold | dual (beta only) | dual (matched) | single (matched) | "
         "margin, beta only | margin, matched |",
         "|---|---|---|---|---|---|"]

    for thr in THRESH:
        c_beta, _ = best(D, thr, (BETA,))
        c_match, _ = best(D, thr, (BETA, "gain=", "qmap"))
        singles = [best(s, thr, ("gain=", "qmap"))[0] for s in S]
        singles = [c for c in singles if c is not None]
        if c_beta is None or c_match is None or not singles:
            continue
        m = sum(singles) / len(singles)
        L.append(f"| {thr} | {c_beta:.4f} | {c_match:.4f} | {m:.4f} | "
                 f"{c_beta - m:+.4f} | {c_match - m:+.4f} |")

    L.append("\nThe differential carries the result. The same two checkpoints, "
             "the same test set and the same selection rule are held fixed. "
             "Only whether both sides may use the knob family changes, and the "
             "margin moves with it.")
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
