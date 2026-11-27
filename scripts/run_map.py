#!/usr/bin/env python3
r"""Six-panel map of an inference run: start, fit, fit minus start, Laplace sd, reflectivity, and a reference.

Usage: python scripts/run_map.py RESULT.npz PASSES.npz [--start START.npz] [--truth TRUTH.npz | --reference-3dss POINTS.npy]
       [--title TEXT] [--out figs/<stem>_map.png]

RESULT.npz is a state `scripts/infer.py` wrote, START.npz its start (default `<stem>_start.npz` next to it),
PASSES.npz the passes it was fitted on (`legs.py` output, read only for the tracks). Every vertex within sonar
reach of a track is drawn. The reflectivity panel is the ratio match's per-cell log reflectivity
:math:`\log\hat\kappa_\mathcal{H}` at the final depths (saved by the run, `bbf.match.ratio.cell_reflectivity`),
drawn on the cells themselves, or the field :math:`\xi` of a field run. The sixth panel is the reference, on the
depth panels' color scale:

- `--truth`: the seabed a twin's intensities were drawn from (a saved state, e.g. `runs/shore_day0707.npz`);
  the error rms of the start and of the fit against it is printed;
- `--reference-3dss`: world-frame PingDSP 3DSS points (m, 3), display only: nothing of it reaches the method.

Also prints the change by distance from the tracks, the median sd and the field's spread.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from bbf.io import load_depths


def tracks_of(passes_file: Path) -> list[np.ndarray]:
    """Horizontal tracks (n, 2) of every pass in a `legs.py` passes file."""
    z = np.load(passes_file)
    n = len({k.split("/", 1)[0] for k in z.files if "/" in k})
    return [z[f"{i}/poses"][:, :2] for i in range(n)]


def reach_of(passes_file: Path) -> float:
    """Sonar reach (m) of the first pass: bins times bin size."""
    z = np.load(passes_file)
    return float(z["0/y"].shape[-1] * float(z["0/dr"]))


def cell_panel(fig, ax, mesh, ex: dict) -> None:
    r"""The ratio match's cells colored by :math:`\log\hat\kappa_\mathcal{H}` minus its median: ring cells as their faces,
    square cells as squares of side `cell`."""
    lk = ex["log_kappa_cell"] - np.median(ex["log_kappa_cell"])
    key = np.asarray(ex["cell_key"])
    lo, hi = np.percentile(lk, [1, 99])
    lim = max(abs(lo), abs(hi))
    origin = np.asarray(mesh.origin)
    if key.shape[1] == 1:  # ring cells: the key is the patch of the faces
        from matplotlib.tri import Triangulation
        from bbf.mesh import ring_patches
        patch = np.asarray(ring_patches(mesh))
        value = np.full(int(patch.max()) + 1, np.nan)
        value[key[:, 0]] = lk
        fv = value[patch]
        v = mesh.vertices + origin
        tri = Triangulation(v[:, 0], v[:, 1], mesh.triangles)
        tri.set_mask(np.isnan(fv))
        h = ax.tripcolor(tri, facecolors=np.nan_to_num(fv), cmap="gray", vmin=-lim, vmax=lim, rasterized=True)
        kind = "one-ring cells"
    else:
        cell = float(ex["cell"]) if "cell" in ex else 1.0
        c = (key + 0.5) * cell + origin
        h = ax.scatter(c[:, 0], c[:, 1], c=lk, s=1, marker="s", cmap="gray", vmin=-lim, vmax=lim, rasterized=True)
        kind = f"{cell:g} m cells"
    fig.colorbar(h, ax=ax, label="log reflectivity minus median")
    ax.set_title(f"cell log reflectivity at the fit ({kind}; integrated out in the fit)")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("result", type=Path)
    ap.add_argument("passes_file", type=Path, metavar="PASSES.npz")
    ap.add_argument("--start", type=Path, help="the run's start (default <stem>_start.npz)")
    ref = ap.add_mutually_exclusive_group()
    ref.add_argument("--truth", type=Path, help="twin truth: the state whose seabed the twin was drawn from")
    ref.add_argument("--reference-3dss", type=Path, dest="points", help="world 3DSS points (m, 3), display only")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    stem = args.result.stem
    start = args.start or args.result.with_name(f"{stem}_start.npz")
    mesh, d, ex = load_depths(args.result)
    _, d0, _ = load_depths(start)
    tr = tracks_of(args.passes_file)
    v = mesh.vertices + np.asarray(mesh.origin)
    dist = KDTree(np.concatenate(tr)).query(v, distance_upper_bound=reach_of(args.passes_file))[0]
    near = dist < np.inf
    ch, sd = d - d0, ex["sd_d"]
    xi = ex["xi"] if "xi" in ex else np.zeros_like(d)
    dt = load_depths(args.truth)[1] if args.truth else None

    print(f"{stem}: fit - start median {np.median(ch[near]):+.3f} m, 5-95% {np.percentile(ch[near], 5):+.2f}.."
          f"{np.percentile(ch[near], 95):+.2f}, |change| > 0.5 m on {np.mean(np.abs(ch[near]) > 0.5):.1%}")
    for lo, hi in ((0, 5), (5, 20), (20, 51)):
        k = near & (dist >= lo) & (dist < hi)
        print(f"  {lo}-{hi} m from a track: depth 5-95% {np.percentile(d[k], 5):.1f}..{np.percentile(d[k], 95):.1f} m, "
              f"median sd {np.median(sd[k]):.2f} m, median |change| {np.median(np.abs(ch[k])):.3f} m")
    if "log_kappa_cell" in ex:
        lk, sp = ex["log_kappa_cell"], ex["cell_spread"]
        print(f"  cells: {len(lk)}, log reflectivity sd {lk.std():.2f}, 1-99% {np.percentile(lk, 1):+.1f}.."
              f"{np.percentile(lk, 99):+.1f}; spread of visits median {np.median(sp):.2f}, 90% {np.percentile(sp, 90):.2f}")
    else:
        print(f"  xi: sd {xi[near].std():.2f}, 1-99% {np.percentile(xi[near], 1):+.1f}..{np.percentile(xi[near], 99):+.1f}"
              + ("" if "xi" in ex else " (no field)"))

    lim = np.percentile(np.concatenate([d0[near], d[near]] + ([dt[near]] if dt is not None else [])), [1, 99])
    cells = "log_kappa_cell" in ex
    panels = [(d0, "start: depth below z_ref (m)", "viridis_r", lim), (d, "fit: depth below z_ref (m)", "viridis_r", lim),
              (ch, "fit minus start (m)", "RdBu", (-2, 2)),
              (sd, "Laplace standard deviation (m)", "magma", np.percentile(sd[near], [1, 99])),
              (xi, "log reflectivity (field)", "gray", (-4.6, 5.7))]
    fig, axs = plt.subplots(2, 3, figsize=(22, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, (val, title, cmap, (lo, hi)) in zip(axs, panels[:4] if cells else panels):
        sc = ax.scatter(v[near, 0], v[near, 1], c=val[near], s=1, cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
        fig.colorbar(sc, ax=ax, label=title)
        ax.set_title(title)
    if cells:
        cell_panel(fig, axs[4], mesh, ex)
    ax = axs[5]
    if dt is not None:
        sc = ax.scatter(v[near, 0], v[near, 1], c=dt[near], s=1, cmap="viridis_r", vmin=lim[0], vmax=lim[1], rasterized=True)
        ax.set_title(f"twin truth ({args.truth.stem}), same color scale")
        rms = lambda e: np.sqrt(np.mean(e[near] ** 2))
        print(f"  twin truth: error rms start {rms(d0 - dt):.2f} m -> fit {rms(d - dt):.2f} m (all vertices in reach)")
    elif args.points is not None:
        pts = np.load(args.points)[::3]
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=mesh.z_ref - pts[:, 2], s=0.1, cmap="viridis_r", vmin=lim[0], vmax=lim[1],
                        rasterized=True)
        ax.set_title("3DSS reference (not used by the method), same color scale")
    else:
        sc = None
        ax.set_axis_off()
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="depth below z_ref (m)")
    for ax, (_, _, cmap, _) in zip(axs, panels + [(None, None, "viridis_r", None)]):
        for p in tr:
            ax.plot(p[:, 0], p[:, 1], "k" if cmap == "RdBu" else "w", lw=0.6)
        ax.set_aspect("equal")
    fig.suptitle(f"{args.result}: {args.title}", fontsize=10)
    fig.tight_layout()
    out = args.out or Path("figs") / f"{stem}_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
