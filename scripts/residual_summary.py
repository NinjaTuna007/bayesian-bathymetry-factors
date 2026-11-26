#!/usr/bin/env python3
r"""Summary figure of the model candidates: :math:`\psi` step by step, and the held-out scores.

Usage: python scripts/residual_summary.py --psi runs/psi_*.json --scores figs/residuals/residual_models_*.json [--out figs/residual_summary.png]

Top row: the beam centre, beam width and linear gain of each side after every Gauss--Newton step
of the runs `scripts/residual_psi_gn.py` wrote, one line per run and side (port solid, starboard
dashed), with the bounds the optimiser keeps :math:`\psi` within. Bottom row: for each score file of
`scripts/residual_models.py`, the held-out log likelihood per bin of every candidate against the
file's first, and the standard deviation of the difference between the two passes' fields.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#3a9a5c", "#52514e", "#8a4fc7", "#c7a11f", "#d6457a", "#2aa5a0"]


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--psi", type=Path, nargs="*", default=[])
    ap.add_argument("--scores", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("figs/residual_summary.png"))
    args = ap.parse_args()
    n_s = max(len(args.scores), 1)
    fig = plt.figure(figsize=(21, 6 + 0.42 * max([len(json.loads(f.read_text())) for f in args.scores] + [6]) + 1))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
    axp = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    titles = ("Beam centre (deg below the outboard axis)", "Beam width (deg)", "Linear gain (dB/m)")
    bounds = ((5.0, 85.0), (10.0, 120.0), (-1.0, 2.0))
    for f, colr in zip(args.psi, COLORS):
        run = json.loads(f.read_text())
        psi = np.array(run["psi"])
        for k, ax in enumerate(axp):
            v = np.degrees(psi[:, :, k]) if k < 2 else psi[:, :, k]
            ax.plot(v[:, 0], "-o", ms=3, color=colr, label=f.stem.removeprefix("psi_"))
            ax.plot(v[:, 1], "--s", ms=3, color=colr)
    for ax, title, (lo, hi) in zip(axp, titles, bounds):
        ymin, ymax = ax.get_ylim()
        for b in (lo, hi):
            if ymin - 0.2 * (ymax - ymin) < b < ymax + 0.2 * (ymax - ymin):
                ax.axhline(b, color="k", lw=0.8, ls=":")
        ax.set(xlabel="accepted Gauss-Newton step", title=title + "; port solid, starboard dashed")
        ax.spines[["top", "right"]].set_visible(False)
    axp[0].legend(frameon=False, fontsize=8)
    for i, f in enumerate(args.scores):
        res = json.loads(f.read_text())
        names = [r["model"] for r in res]
        yy = np.arange(len(res))
        w = 6 // n_s
        ax = fig.add_subplot(gs[1, i * w:(i + 1) * w])
        dll = np.array([r["ll_test"] - res[0]["ll_test"] for r in res])
        ax.barh(yy, dll, color=COLORS[0], height=0.6)
        for y, r, v in zip(yy, res, dll):
            ax.text(max(v, 0) + 0.002, float(y), f"{v:+.3f}   d-xi {r['dxi_sd']:.2f}   transfer {r['transfer_A']:+.2f} / {r['transfer_B']:+.2f}",
                    va="center", fontsize=8)
        ax.set_yticks(yy, names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        span = max(np.abs(dll).max(), 0.02)
        ax.set_xlim(min(dll.min(), 0) - 0.05 * span, max(dll.max(), 0) + 1.6 * span)
        ax.set(xlabel=f"held-out log likelihood per bin against {names[0]}", title=f.stem.removeprefix("residual_models_"))
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
