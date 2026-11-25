#!/usr/bin/env python3
r"""The horizontal offset between two passes, without the radiometric model, and what could cause it.

Usage: python scripts/residual_offset.py PATCHES.npz [--H 21.4] [--out figs/residual_offset.png]

As `scripts/residual_geometry.py`: every bin beyond the first echo is put on a level seabed
:math:`H` below the reference height, its log intensity minus a 3 m running mean along range is
averaged over ground cells of 0.5 m, and the two passes' maps are correlated over the cells both
cover, for the pair of sides that looks the same way (A's port and B's starboard).

1. The correlation against a horizontal move of pass B's positions, :math:`(\Delta x, \Delta y)`: its
   peak is the offset to give B (`bbf.io.Pass.shifted`), and its width the precision.
2. The same peak per third of pass B along track and per band of B's range: an offset of the
   navigation is one vector everywhere; an error in depth, pitch, yaw or timing moves the
   footprints by amounts that vary with range or with the heading.
3. With candidate causes applied to both passes instead (a pitch offset, a shift of the ping
   times, a yaw offset), the best move of B that is left: a cause that explains the offset leaves none.

Only the intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE
from residual_geometry import CELL, cell_means, footprints, shift_search, texture

from bbf.inference import init as I
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import load_patch
from bbf.ping import Theta, bin_ranges

PAIR = (0, 1)  # A's port and B's starboard look the same way


def corr(a, b) -> tuple[float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    return (float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 500 else np.nan), int(m.sum())


def surface(map_a, q_b, v_b, origin, shape, moves) -> np.ndarray:
    """Correlation of A's map with B's bins moved by each (east, north) of `moves` (n, n, 2), a grid of whole cells about zero.

    Moving B's bins by :math:`m` moves its map's content by :math:`m`, which is `shift_search`'s shift :math:`-m`.
    """
    surf, _ = shift_search(map_a, cell_means(q_b, v_b, origin, shape), max_shift=(moves.shape[0] - 1) // 2)
    return surf[::-1, ::-1]


def peak(surf, moves) -> tuple[np.ndarray, float]:
    """Move at the surface's peak, refined by a parabola through its neighbours, and the peak value."""
    i, j = np.unravel_index(np.nanargmax(surf), surf.shape)
    best = moves[i, j].astype(float).copy()
    for ax, (k, n) in enumerate(((i, surf.shape[0]), (j, surf.shape[1]))):
        if 0 < k < n - 1:
            c = [surf[(k + s, j) if ax == 0 else (i, k + s)] for s in (-1, 0, 1)]
            den = c[0] - 2 * c[1] + c[2]
            if np.isfinite(den) and den < 0:
                step = moves[1, 0, 0] - moves[0, 0, 0] if ax == 0 else moves[0, 1, 1] - moves[0, 0, 1]
                best[ax] += 0.5 * (c[0] - c[2]) / den * step
    return best, float(surf[i, j])


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--H", type=float, default=21.4, help="level depth below the reference height (m)")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path("figs/residual_offset.png"))
    args = ap.parse_args()
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    full = [load_patch(args.patches, n) for n in "AB"]
    z_ref = float(np.median(np.concatenate([p.poses[:, 2] for p in full])))
    passes = [coarsen_pass(p, args.k) for p in full]
    th = coarsen_theta(theta, args.k)
    tex = [texture(p, I.first_echo(p, th)) for p in passes]
    xy = np.concatenate([p.poses[:, :2] for p in passes])
    origin = xy.min(axis=0) - 55.0
    shape = tuple(int(n) for n in np.ceil((xy.max(axis=0) + 55.0 - origin) / CELL))
    assert len(shape) == 2
    g = np.arange(-12, 13) * CELL
    moves = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1)
    ca, cb = PAIR

    def maps(pitch=0.0, dt=0.0, yaw=0.0):
        qa = footprints(passes[0], args.H, z_ref, pitch, dt, yaw)[:, ca]
        qb = footprints(passes[1], args.H, z_ref, pitch, dt, yaw)[:, cb]
        return cell_means(qa, tex[0][:, ca], origin, shape), qb

    map_a, qb = maps()
    vb = tex[1][:, cb]
    surf = surface(map_a, qb, vb, origin, shape, moves)
    best, val = peak(surf, moves)
    zero = surf[len(g) // 2, len(g) // 2]
    half = np.argwhere(surf > 0.5 * (val + np.nanmedian(surf)))
    print(f"level seabed at {args.H} m: correlation {zero:+.3f} with B where the navigation has it; peak {val:+.3f} with B's "
          f"positions moved by ({best[0]:+.2f}, {best[1]:+.2f}) m east, north; half-height region "
          f"{g[half[:, 0].min()]:+.1f}..{g[half[:, 0].max()]:+.1f} m east, {g[half[:, 1].min()]:+.1f}..{g[half[:, 1].max()]:+.1f} m north")
    hb = np.degrees(np.median(passes[1].poses[:, 5]))
    ha = np.degrees(passes[0].poses[[0, -1], 5])
    al = np.array([np.sin(np.radians(hb)), np.cos(np.radians(hb))])
    print(f"  B heads {hb:.0f} deg (A {ha[0]:.0f} to {ha[1]:.0f}): the move is {best @ al:+.2f} m along B's track and "
          f"{best @ np.array([al[1], -al[0]]):+.2f} m to its starboard")

    # 2. by part of pass B and by its range
    n = passes[1].n_pings
    r = np.asarray(bin_ranges(passes[1].n_bins, passes[1].dr))
    fine = g
    mv = np.stack(np.meshgrid(fine, fine, indexing="ij"), axis=-1)
    for label, sel_t, sel_r in ([(f"pings {a}..{b}", slice(a, b), slice(None)) for a, b in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))]
                                + [(f"range {a:.0f}..{b:.0f} m", slice(None), (r >= a) & (r < b)) for a, b in ((21, 28), (28, 36), (36, 51))]):
        v = np.full(vb.shape, np.nan)
        v[sel_t, sel_r] = vb[sel_t, sel_r]
        s = surface(map_a, qb, v, origin, shape, mv)
        if np.isfinite(s).sum() < s.size // 2:
            print(f"  B {label}: too few shared cells")
            continue
        bst, vl = peak(s, mv)
        print(f"  B {label}: peak {vl:+.3f} at ({bst[0]:+.2f}, {bst[1]:+.2f}) m; at zero {s[len(fine) // 2, len(fine) // 2]:+.3f}")

    # 3. candidate causes applied to both passes: what move of B is left
    for label, kw in ([(f"pitch offset {x:+.1f} deg", {"pitch": np.radians(x)}) for x in (-6.0, -3.0, 3.0, 6.0)]
                      + [(f"ping times {x:+.2f} s", {"dt": x}) for x in (-1.5, -0.75, 0.75, 1.5)]
                      + [(f"yaw offset {x:+.1f} deg", {"yaw": np.radians(x)}) for x in (-6.0, 6.0, 11.0)]):
        ma, q2 = maps(**kw)
        s = surface(ma, q2, vb, origin, shape, mv)
        bst, vl = peak(s, mv)
        print(f"  {label}: at zero {s[len(fine) // 2, len(fine) // 2]:+.3f}; peak {vl:+.3f} with B moved by ({bst[0]:+.2f}, {bst[1]:+.2f}) m")
    # check: B moved by the estimate
    moved = dataclasses.replace(passes[1], poses=passes[1].shifted(*best).poses)
    q3 = footprints(moved, args.H, z_ref)[:, cb]
    c3, cells_used = corr(map_a, cell_means(q3, vb, origin, shape))
    print(f"with B moved by the estimate: correlation {c3:+.3f} over {cells_used} cells")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(surf.T, origin="lower", extent=(g[0] - 0.25, g[-1] + 0.25, g[0] - 0.25, g[-1] + 0.25), cmap="RdBu_r", vmin=-0.25, vmax=0.25)
    fig.colorbar(im, ax=ax, label="correlation of the passes' textures")
    ax.plot(*best, "k+", ms=12)
    ax.plot(0, 0, "ko", ms=4)
    ax.set(xlabel="move of pass B east (m)", ylabel="move of pass B north (m)",
           title=f"A port against B starboard, level seabed at {args.H} m")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
