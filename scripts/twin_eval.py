#!/usr/bin/env python3
r"""Score runs against a known seabed: error by distance from the tracks, NEES, and a figure.

Usage: python scripts/twin_eval.py TRUTH.npz|POINTS.npy PASSES.npz RUN.npz [RUN.npz ...] [--seen SEEN.npz] [--nees]
       [--figure figs/<stem>_vs_truth.png]

The reference is the twin's truth (a state), or for real data the PingDSP 3DSS bathymetry as world points
POINTS.npy (m, 3): their depths below the run's :math:`z_\mathrm{ref}` are averaged per mesh vertex over the points
nearest it (within `REF_RADIUS` times the mesh side, at least `REF_MIN` points); vertices without enough points
are left out of every band. The 3DSS only scores results here; nothing of it reaches the method.

TRUTH.npz is the state whose depths (and field) made the twin's intensities, e.g. `runs/shore_day0707.npz`;
each RUN.npz a state `scripts/infer.py` wrote on the twin, with its start `<stem>_start.npz` next to it.
Vertices are split into bands by distance to the nearest track and into seen and unseen ground: seen
when a scored bin of the fit lies over it, read from SEEN.npz (`count` per vertex; default
`<stem>_seen.npz` of each run, else of `--seen`; without one, every vertex counts as seen). Per band:
the truth's median depth, the share seen, the error rms on seen ground of start and fit (and the fit's
median :math:`|d - d_\mathrm{true}|`), the same rms on unseen ground, and the median Laplace sd on each.

With `--nees`, a second table per run: the per-vertex normalized squared error
:math:`e_i = (d_i - d_{\mathrm{true},i})^2 / \sigma_i^2` with :math:`\sigma_i` the Laplace marginal sd, by band on
seen and unseen ground: its mean and median, and the share of vertices with :math:`|d_i - d_{\mathrm{true},i}| \le
1.96\,\sigma_i`. A consistent posterior has mean 1, median 0.45 and coverage 95 %. Only the marginal sds are
saved, so this is the per-vertex (scalar) NEES, not the joint NEES of a set of vertices.

"Seen" is taken at the fit's depths: where the fit is too deep its rays reach ground a real bank would hide,
so the seen share is optimistic on wrong seabeds.

With `--figure`, a 3 x 3 figure of the first run: truth, start, fit; start, fit and (second run, if
given) errors against the truth; fit minus start, the seen mask and the Laplace sd.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import KDTree

from bbf.io import load_depths
from run_map import tracks_of

REF_RADIUS, REF_MIN = 0.6, 3
BANDS = ((0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50))


def rms(a):
    return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")


def reference_on_mesh(points: np.ndarray, mesh) -> np.ndarray:
    """Depth (n,) of the reference points averaged per nearest vertex; NaN where fewer than `REF_MIN` lie near."""
    v = mesh.vertices + np.asarray(mesh.origin)
    dist, near = KDTree(v).query(points[:, :2], distance_upper_bound=REF_RADIUS * mesh.h)
    ok = np.isfinite(dist)
    n = np.bincount(near[ok], minlength=len(v))
    s = np.bincount(near[ok], mesh.z_ref - points[ok, 2], minlength=len(v))
    return np.where(n >= REF_MIN, s / np.maximum(n, 1), np.nan)


def load_run(path: Path, fallback_seen: Path | None, n: int):
    """Start and fit depths, Laplace sd, and the seen mask of a run."""
    _, d0, _ = load_depths(path.with_name(f"{path.stem}_start.npz"))
    _, d, ex = load_depths(path)
    own = path.with_name(f"{path.stem}_seen.npz")
    src = own if own.exists() else fallback_seen
    seen = np.load(src)["count"] > 0 if src is not None else np.ones(n, bool)
    return d0, d, ex["sd_d"], seen, src


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("truth", type=Path)
    ap.add_argument("passes_file", type=Path, metavar="PASSES.npz")
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--seen", type=Path, help="seen counts for runs without their own <stem>_seen.npz")
    ap.add_argument("--nees", action="store_true", help="also the per-vertex depth NEES and 95 %% coverage by band")
    ap.add_argument("--figure", type=Path, nargs="?", const=True, help="write the figure (default figs/<stem>_vs_truth.png)")
    args = ap.parse_args()

    if args.truth.suffix == ".npy":
        mesh = load_depths(args.runs[0])[0]
        dt = reference_on_mesh(np.load(args.truth), mesh)
        print(f"reference: 3DSS points on {np.isfinite(dt).sum()} of {len(dt)} vertices")
    else:
        mesh, dt, _ = load_depths(args.truth)
    tr = np.concatenate(tracks_of(args.passes_file))
    v = mesh.vertices + np.asarray(mesh.origin)
    dist = KDTree(tr).query(v)[0]
    dist[~np.isfinite(dt)] = np.inf  # no reference there: out of every band
    runs = {}
    for path in args.runs:
        d0, d, sd, seen, src = runs[path] = load_run(path, args.seen, len(v))
        print(f"\n{path.stem} (seen from {src.name if src else 'none: all vertices'})")
        print("band     | truth med | seen  | seen: start -> fit rms (fit med |err|) | unseen: start -> fit | sd seen / unseen")
        for lo, hi in BANDS:
            k = (dist >= lo) & (dist < hi)
            s, u = k & seen, k & ~seen
            sd_u = f"{np.median(sd[u]):.2f}" if u.any() else "  - "
            print(f"{lo:2d}-{hi:2d} m  | {np.median(dt[k]):6.1f} m | {s.sum() / k.sum():5.1%} | "
                  f"{rms((d0 - dt)[s]):5.2f} -> {rms((d - dt)[s]):5.2f} ({np.median(np.abs(d - dt)[s]):.2f}) | "
                  f"{rms((d0 - dt)[u]):5.2f} -> {rms((d - dt)[u]):5.2f} | {np.median(sd[s]):.2f} / {sd_u}")
        s = (dist < 50) & seen
        print(f"all seen < 50 m: {rms((d0 - dt)[s]):.2f} -> {rms((d - dt)[s]):.2f} m rms; "
              f"error removed {1 - rms((d - dt)[s]) / rms((d0 - dt)[s]):.0%}")
        if args.nees:
            e = (d - dt) ** 2 / sd**2
            cover = np.abs(d - dt) <= 1.96 * sd
            print("NEES     | seen: mean / median / 95% coverage   | unseen: mean / median / 95% coverage")
            for lo, hi in BANDS + ((0, 50),):
                k = (dist >= lo) & (dist < hi)
                cols = []
                for m in (k & seen, k & ~seen):
                    cols.append(f"{e[m].mean():9.3g} / {np.median(e[m]):9.3g} / {cover[m].mean():6.1%}" if m.any()
                                else f"{'-':>9} / {'-':>9} / {'-':>6}")
                print(f"{lo:2d}-{hi:2d} m  | {cols[0]} | {cols[1]}")

    if args.figure:
        first = args.runs[0]
        d0, d, sd, seen, _ = runs[first]
        out = Path("figs") / f"{first.stem}_vs_truth.png" if args.figure is True else args.figure
        tri = Triangulation(v[:, 0], v[:, 1], mesh.triangles)
        tri.set_mask((dist[mesh.triangles] > 50).any(axis=1))
        fig, ax = plt.subplots(3, 3, figsize=(21, 15), sharex=True, sharey=True)
        lo, hi = np.percentile(dt[dist < 50], [1, 99])
        depth = "depth below z_ref (m)"

        def panel(a, val, title, cmap, vmin, vmax, label):
            h = a.tripcolor(tri, val, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
            a.plot(tr[::20, 0], tr[::20, 1], ",", color="w" if cmap == "viridis_r" else "k", alpha=0.5)
            a.set_title(title)
            a.set_aspect("equal")
            fig.colorbar(h, ax=a, shrink=0.8, label=label)

        panel(ax[0, 0], dt, f"truth ({args.truth.stem})", "viridis_r", lo, hi, depth)
        panel(ax[0, 1], d0, "start", "viridis_r", lo, hi, depth)
        panel(ax[0, 2], d, f"fit: {first.stem}", "viridis_r", lo, hi, depth)
        panel(ax[1, 0], d0 - dt, "start minus truth", "RdBu", -12, 12, "m (red: too shallow, blue: too deep)")
        panel(ax[1, 1], d - dt, f"{first.stem} minus truth", "RdBu", -12, 12, "m")
        if len(args.runs) > 1:
            other = args.runs[1]
            panel(ax[1, 2], runs[other][1] - dt, f"{other.stem} minus truth", "RdBu", -12, 12, "m")
        else:
            ax[1, 2].set_axis_off()
        panel(ax[2, 0], d - d0, f"{first.stem}: fit minus start", "RdBu", -6, 6, "m")
        panel(ax[2, 1], seen.astype(float), "seen ground (under a scored bin, at the fit's depths)", "Greys", 0, 1.5, "seen")
        panel(ax[2, 2], np.log10(sd), f"{first.stem}: Laplace sd", "magma", -1.5, 1, "log10 sd (m)")
        fig.suptitle(f"Twin: {first.stem} against the known seabed (within 50 m of the tracks)", fontsize=14)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=80)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
