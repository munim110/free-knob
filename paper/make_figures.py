"""Figures for the paper, generated from the same JSONs as the tables.

fig_illustration.pdf
    one observed frame and a mean-error-trained forecast of it, then both after
    a max over 16x16 blocks and a threshold. Drawn from a cached artefact so the
    figure rebuilds without the prediction HDF5s.

fig_mechanism.pdf
    (a) pooled frequency bias against threshold, per system. Max-pooling drives
        smooth forecasts' bias down while leaving a real radar field at one.
    (b) the resulting dose-response: how much a free monotone knob buys, against
        how far below one the bias sits. One point per (system, threshold,
        pooling).

fig_decomposition.pdf
    two comparisons at the most extreme cell, each as reported and after the
    identical global quantile map: (a) two released checkpoints of one
    architecture, (b) the CasCast cascade against its backbone.

fig_contrasts.pdf
    every pairwise contrast: calibration gap against distortion, and raw
    against calibrated contrast, with sign reversals marked.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

RES = Path(str(P.RESULTS))
OUT = Path(str(P.PAPER))
THR = [16, 74, 133, 160, 181, 219]
POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]

SYS = [("persistence", "persistence (real field)", "#1b7837", "o"),
       ("pysteps", "pysteps LK", "#5aae61", "s"),
       ("earthformer", "EarthFormer", "#f1a340", "^"),
       ("cascast_det", "CasCast backbone", "#d73027", "v"),
       ("cascast_cascade", "CasCast cascade", "#762a83", "D")]


def load(name):
    f = RES / f"sevir_poolingq_{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {(r["pooling"], r["threshold"]): r for r in d["table"]}


def schematic(fig, gs):
    """The mechanism as a picture, from one real event rather than a cartoon.

    Left pair: the observed field and a mean-error-trained forecast of it, same
    colour scale. Right pair: the same two after a max over 16x16 blocks and a
    threshold at 219. The observation keeps its blocks; the forecast loses
    nearly all of them, without having put anything in the wrong place.
    """
    f = RES / "sevir_schematic.npz"
    if not f.exists():
        return False
    z = np.load(f, allow_pickle=True)
    k = int(z["pool"])
    thr = int(z["threshold"])
    panels = [(z["gt"], "observed field", "field"),
              (z["pred"], f"forecast ({z['arm']})", "field"),
              (z["gt_mask"], rf"observed, max ${k}^2$, $\geq{thr}$", "mask"),
              (z["pred_mask"], rf"forecast, max ${k}^2$, $\geq{thr}$", "mask")]
    for i, (img, title, kind) in enumerate(panels):
        a = fig.add_subplot(gs[i])
        if kind == "field":
            a.imshow(img, cmap="magma", vmin=0, vmax=255,
                     interpolation="nearest", aspect="equal")
        else:
            a.imshow(img, cmap="Greys", vmin=0, vmax=1,
                     interpolation="nearest", aspect="equal")
            # draw the block lattice so the panel reads as pooled cells rather
            # than as a low-resolution image
            n = img.shape[0]
            a.set_xticks(np.arange(-0.5, n, 1), minor=True)
            a.set_yticks(np.arange(-0.5, n, 1), minor=True)
            a.grid(which="minor", color="0.75", lw=0.25)
            a.tick_params(which="minor", length=0)
            a.set_xlabel(f"{int(img.sum())} of {img.size} blocks", fontsize=6.5,
                         labelpad=1.5)
        a.set_title(title, fontsize=7, pad=2)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    return True


def illustration():
    if not (RES / "sevir_schematic.npz").exists():
        print("  (illustration skipped: sevir_schematic.npz missing)")
        return
    plt.rcParams.update({"font.size": 8, "axes.grid": False})
    fig = plt.figure(figsize=(6.9, 1.75))
    schematic(fig, fig.add_gridspec(1, 4, wspace=0.1))
    fig.savefig(
        OUT / "fig_illustration.pdf", bbox_inches="tight",
        metadata={"Creator": "freeknob",
                  "Producer": "Matplotlib", "CreationDate": None},
    )
    print("  wrote fig_illustration.pdf")


def mechanism():
    # drawn close to the \textwidth it is included at (~5.5in), so the text does
    # not get shrunk by the scale factor on the way into the page
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.3))

    # (a) bias against threshold under the reporting convention used for
    #     extreme-event claims. A real radar field holds at one; everything
    #     trained on a mean error collapses as the threshold rises.
    xs = np.arange(len(THR))
    for name, label, c, mk in SYS:
        T = load(name)
        if not T:
            continue
        y = [T[("max16", t)]["bias_uncal"] for t in THR]
        ax[0].plot(xs, y, marker=mk, color=c, label=label, lw=1.8, ms=5)
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([str(t) for t in THR])
    ax[0].set_xlabel(r"threshold $\tau$ (VIL)")
    ax[0].set_ylabel("pooled frequency bias")
    ax[0].set_title(r"(a) pooled frequency bias by threshold, max $16^2$",
                    fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")
    ax[0].set_ylim(0, 1.35)

    # (b) dose-response over every (system, threshold, pooling) cell
    for name, label, c, mk in SYS:
        T = load(name)
        if not T:
            continue
        bx = [T[(pl, t)]["bias_uncal"] for pl in POOLINGS for t in THR]
        by = [100 * T[(pl, t)]["rel_gain_qmap_global"]
              for pl in POOLINGS for t in THR]
        ax[1].scatter(bx, by, color=c, marker=mk, s=30, alpha=0.78,
                      edgecolors="none", label=label)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(1.0, color="k", ls=":", lw=1)
    ax[1].set_xlabel("pooled frequency bias of the released model")
    ax[1].set_ylabel("CSI gain from one global\nquantile map (%)")
    ax[1].set_title("(b) calibration gain against frequency bias",
                    fontsize=8.5)
    ax[1].set_xlim(0, 1.7)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(
        OUT / "fig_mechanism.pdf", bbox_inches="tight",
        metadata={"Creator": "freeknob",
                  "Producer": "Matplotlib", "CreationDate": None},
    )
    print("  wrote fig_mechanism.pdf")


def decomposition():
    D, C = load("cascast_det"), load("cascast_cascade")
    f = RES / "sevir_published_contrasts.json"
    if not (D and C and f.exists()):
        print("  (decomposition figure skipped: inputs not available)")
        return
    pl, t = "max16", 219
    # Same sources as the macros quoted next to the figure: SameArch* is read
    # from the contrast file, Eight* from the per-system pooling files.
    sa = [r for r in json.loads(f.read_text())["rows"]
          if r.get("same_arch") and r["pooling"] == pl and r["threshold"] == t][0]
    same = ([sa["csi_a"], sa["csi_b"]], [sa["csi_a_cal"], sa["csi_b_cal"]])
    casc = ([D[(pl, t)]["csi_uncal"], C[(pl, t)]["csi_uncal"]],
            [D[(pl, t)]["csi_qmap_global"], C[(pl, t)]["csi_qmap_global"]])

    panels = [
        ("(a) One architecture, two released checkpoints", same,
         ["EarthFormer release", "CasCast backbone"], ["#f1a340", "#d73027"]),
        ("(b) A published gain: cascade against its backbone", casc,
         ["CasCast backbone", "CasCast cascade"], ["#d73027", "#762a83"]),
    ]

    # included at \textwidth (~5.5in), so the text is drawn close to final size
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.grid": False,
                         "xtick.labelsize": 7, "ytick.labelsize": 6.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.2), sharey=True)
    out = []
    for ax, (title, (raw, cal), names, cols) in zip(axes, panels):
        top = max(raw + cal)
        for j in range(2):
            b = ax.bar([j * 0.36 - 0.18, 1 + j * 0.36 - 0.18], [raw[j], cal[j]],
                       width=0.32, color=cols[j], label=names[j])
            ax.bar_label(b, fmt="%.3f", fontsize=6, padding=1.5)
        for x, v in ((0, raw), (1, cal)):
            ax.annotate(f"difference {(v[1] - v[0]) / v[0]:+.1%}",
                        xy=(x, 0.385), ha="center", fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["raw score", "after the same calibration"])
        ax.set_ylim(0, 0.44)
        ax.set_title(title, fontsize=7, loc="left", pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, -0.2), handlelength=1.2,
                  columnspacing=1.2)
        out.append((raw[1] - raw[0], cal[1] - cal[0],
                    (raw[1] - raw[0]) / raw[0], (cal[1] - cal[0]) / cal[0]))
    axes[0].set_ylabel(r"$\mathrm{CSI}_{219}$, max $16^2$")
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(
        OUT / "fig_decomposition.pdf", bbox_inches="tight",
        metadata={"Creator": "freeknob",
                  "Producer": "Matplotlib", "CreationDate": None},
    )
    for name, o in zip("ab", out):
        print(f"  fig_decomposition.pdf ({name}): gap {o[0]:+.4f} -> {o[1]:+.4f}, "
              f"relative {o[2]:+.1%} -> {o[3]:+.1%}")
    print("  bars:", [[round(v, 3) for v in q] for _, p, _, _ in panels for q in p])


def contrasts():
    """The pairwise-contrast result: (a) calibration gap against distortion,
    (b) raw against calibrated contrast, sign reversals marked in both."""
    f = RES / "sevir_published_contrasts.json"
    if not f.exists():
        print("  (contrast figure skipped: sevir_published_contrasts.json missing)")
        return
    d = json.loads(f.read_text())
    rows, st = d["rows"], d["stats"]
    keep = [r for r in rows if not r["flip"]]
    flip = [r for r in rows if r["flip"]]

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.grid": False,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.35))
    grey, red = "0.55", "#c1121f"

    ax[0].axhline(0, lw=0.5, c="0.75")
    ax[0].axvline(0, lw=0.5, c="0.75")
    ax[0].scatter([r["devgap"] for r in keep], [r["distort"] for r in keep],
                  s=6, c=grey, lw=0, label=f"sign preserved ($n={len(keep)}$)")
    ax[0].scatter([r["devgap"] for r in flip], [r["distort"] for r in flip],
                  s=10, c=red, lw=0, label=f"sign reversed ($n={len(flip)}$)")
    ax[0].set_xlabel(r"calibration gap, $\mathrm{dev}(A)-\mathrm{dev}(B)$")
    ax[0].set_ylabel(r"distortion (raw $-$ calibrated $\Delta$CSI)")
    ax[0].set_title(f"(a) $r={st['all']['pearson']:+.3f}$ over "
                    f"{st['all']['n']} contrasts", fontsize=7.5)
    ax[0].legend(fontsize=6, frameon=False, loc="upper left")

    lim = 0.30
    ax[1].axhline(0, lw=0.5, c="0.75")
    ax[1].axvline(0, lw=0.5, c="0.75")
    ax[1].plot([-lim, lim], [-lim, lim], lw=0.5, c="0.75", ls="--", zorder=0)
    ax[1].scatter([r["raw"] for r in keep], [r["cal"] for r in keep],
                  s=6, c=grey, lw=0)
    ax[1].scatter([r["raw"] for r in flip], [r["cal"] for r in flip],
                  s=10, c=red, lw=0)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel(r"raw contrast $\Delta$CSI")
    ax[1].set_ylabel(r"calibrated contrast $\Delta$CSI")
    ax[1].set_title("(b) raw against calibrated contrast", fontsize=7.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        OUT / "fig_contrasts.pdf", bbox_inches="tight",
        metadata={"Creator": "freeknob",
                  "Producer": "Matplotlib", "CreationDate": None},
    )
    print(f"  wrote fig_contrasts.pdf ({len(flip)} of {len(rows)} reversed)")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5})
    illustration()
    mechanism()
    decomposition()
    contrasts()
