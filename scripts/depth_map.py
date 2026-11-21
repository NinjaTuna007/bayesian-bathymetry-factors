#!/usr/bin/env python3
r"""Map of saved depths (and Laplace standard deviations, if the file has them) over the swaths.

Usage: python scripts/depth_map.py DEPTHS.npz PATCHES.npz [--patches A B] [--out figs/<stem>_map.png]

Only vertices within sonar range of a ping are drawn; the tracks are overlaid in white.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from bbf.io import load_depths, load_patch


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("depths", type=Path)
    ap.add_argument("patches", type=Path)
    ap.add_argument("--patches-used", nargs="+", default=["A", "B"], dest="names")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    mesh, d, extra = load_depths(args.depths)
    passes = [load_patch(args.patches, n) for n in args.names]
    v = mesh.vertices + np.asarray(mesh.origin)
    tracks = np.concatenate([p.poses[:, :2] for p in passes])
    reach = passes[0].n_bins * passes[0].dr
    near = KDTree(tracks).query(v, distance_upper_bound=reach)[0] < np.inf
    panels = [(d, "depth below z_ref (m)", "viridis_r")]
    if "sd_d" in extra:
        panels.append((extra["sd_d"], "Laplace standard deviation (m)", "magma"))
    fig, axs = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), sharex=True, sharey=True, squeeze=False)
    for ax, (val, title, cmap) in zip(axs[0], panels):
        lo, hi = np.percentile(val[near], [1, 99])
        sc = ax.scatter(v[near, 0], v[near, 1], c=val[near], s=0.5, cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(sc, ax=ax, label=title)
        for p in passes:
            ax.plot(p.poses[:, 0], p.poses[:, 1], "w", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("east (m, patch frame)")
        ax.set_title(f"{args.depths.stem}: {title}")
    axs[0][0].set_ylabel("north (m, patch frame)")
    fig.tight_layout()
    out = args.out or Path("figs") / f"{args.depths.stem}_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
