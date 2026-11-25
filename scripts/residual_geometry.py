#!/usr/bin/env python3
r"""Geometry without the radiometric model: at which depth and pose offsets do two passes' textures line up?

Usage: python scripts/residual_geometry.py PATCHES.npz [--out figs/residual_geometry.png]

Each bin beyond the first echo is put on the ground as if the seabed were level at a distance
:math:`H` below the sonar's reference height: in the ping plane (normal :math:`R e_x`, so the pitch is
kept), at slant range :math:`r`, the point with :math:`z = z_\mathrm{ref} - H`. Its value is the log
intensity minus its running mean over 3 m of range, which removes gain, beam and angular law and
keeps the texture. The texture of each pass and side is averaged over ground cells, and the
correlation between the two passes over the cells both cover is a score of the geometry that
needs no model of the echo: reflectivity, objects and shadows cast the same way line up only
when the footprints are right. Slopes seen from opposite directions shade with opposite sign,
so the pairs of sides are scored separately, and one pair that looks the same way (by default A's
port and B's starboard) steers the sweeps.

Swept: the level depth :math:`H`; an offset added to the pitch of both passes; a shift of the ping
times (along track, opposite for opposite headings); an offset added to the yaw (for parallel
tracks a distance :math:`s` apart it moves the passes' footprints apart by :math:`s` times the offset,
like a horizontal offset between the passes). A level seabed is the crudest surface, so
the peak is the depth of the ground the pair of sides shares, not a map. Last, the correlation
against a horizontal shift of pass B: a peak off zero is an offset between the passes.

Only the intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE
from scipy.ndimage import uniform_filter1d

from bbf.inference import init as I
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import Pass, load_patch
from bbf.ping import Theta, bin_ranges

COLORS = ["#2a78d6", "#eb6834", "#3a9a5c", "#52514e"]
SIDES = ("port", "stbd")
CELL = 0.5


def texture(ps: Pass, r0: np.ndarray, window: float = 3.0) -> np.ndarray:
    """Log intensity minus its running mean along range, (n, 2, B); NaN before the first echo plus 1 m and where y = 0."""
    r = np.asarray(bin_ranges(ps.n_bins, ps.dr))
    ok = (ps.y > 0) & (r[None, None] > np.where(np.isfinite(r0), r0, np.inf)[:, :, None] + 1.0)
    ly = np.where(ok, np.log(np.where(ok, ps.y, 1.0)), 0.0)
    w = max(int(window / ps.dr), 3)
    cnt = uniform_filter1d(ok.astype(float), w, axis=-1, mode="constant")
    mean = uniform_filter1d(ly, w, axis=-1, mode="constant") / np.maximum(cnt, 1e-9)
    return np.where(ok & (cnt > 0.5), ly - mean, np.nan)


def footprints(ps: Pass, H: float, z_ref: float, pitch: float = 0.0, dt: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    r"""Ground points (n, 2, B, 2) of every bin on a level seabed :math:`H` below `z_ref`; NaN where :math:`r` does not reach it.

    pitch, yaw: offsets added to the navigation's pitch and yaw (rad); dt: shift of the ping times
    (s), applied by moving each ping along its track.
    """
    poses = ps.poses.copy()
    poses[:, 4] += pitch
    poses[:, 5] += yaw
    if dt:
        v = np.gradient(poses[:, :2], ps.t, axis=0)
        poses[:, :2] += v * dt
    r = np.asarray(bin_ranges(ps.n_bins, ps.dr))
    heading, pit = poses[:, 5], poses[:, 4]
    # heading :math:`R e_x` in east, north, up (`bbf.ping.geometry.pose_frame`); the roll does not enter
    hd = np.column_stack([np.sin(heading) * np.cos(pit), np.cos(heading) * np.cos(pit), np.sin(pit)])
    fwd = hd[:, :2] / np.linalg.norm(hd[:, :2], axis=1, keepdims=True)
    below = H - (z_ref - poses[:, 2])
    out = np.full((ps.n_pings, 2, ps.n_bins, 2), np.nan)
    for c, sign in ((0, 1.0), (1, -1.0)):
        o = sign * np.column_stack([-fwd[:, 1], fwd[:, 0], np.zeros(len(fwd))])  # horizontal outboard axis, in the ping plane
        w = np.cross(o, hd)
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        w = np.where(w[:, 2:3] > 0, -w, w)  # the plane's other axis, pointing down
        cg = below[:, None] / (r[None] * np.abs(w[:, 2:3]))
        ok = cg <= 1.0
        sg = np.sqrt(np.where(ok, 1.0 - cg**2, np.nan))
        cg = np.where(ok, cg, np.nan)
        out[:, c] = (poses[:, None, :2] + r[None, :, None] * (cg[..., None] * w[:, None, :2] + sg[..., None] * o[:, None, :2]))
    return out


def cell_means(q: np.ndarray, v: np.ndarray, origin: np.ndarray, shape: tuple[int, int], min_count: int = 3) -> np.ndarray:
    """Mean of v over ground cells, (nx, ny); NaN below `min_count` values."""
    ok = np.isfinite(v) & np.isfinite(q).all(axis=-1)
    ij = np.floor((q[ok] - origin) / CELL).astype(int)
    inb = (ij >= 0).all(axis=1) & (ij < np.array(shape)).all(axis=1)
    flat = ij[inb, 0] * shape[1] + ij[inb, 1]
    cnt = np.bincount(flat, minlength=shape[0] * shape[1])
    tot = np.bincount(flat, v[ok][inb], minlength=shape[0] * shape[1])
    return np.where(cnt >= min_count, tot / np.maximum(cnt, 1), np.nan).reshape(shape)


def shift_search(a: np.ndarray, b: np.ndarray, max_shift: int = 30) -> tuple[np.ndarray, int]:
    """Correlation of cell maps a and b(x + shift) over shifts of up to `max_shift` cells each way; NaN under 1000 shared cells."""
    n = 2 * max_shift + 1
    out = np.full((n, n), np.nan)
    pad = np.full((b.shape[0] + 2 * max_shift, b.shape[1] + 2 * max_shift), np.nan)
    pad[max_shift:-max_shift, max_shift:-max_shift] = b
    fa = np.isfinite(a)
    for i in range(n):
        for j in range(n):
            bs = pad[i:i + a.shape[0], j:j + a.shape[1]]
            m = fa & np.isfinite(bs)
            if m.sum() > 1000:
                out[i, j] = np.corrcoef(a[m], bs[m])[0, 1]
    return out, max_shift


def pair_scores(passes, tex, z_ref, H, origin, shape, pitch=0.0, dt=0.0, yaw=0.0) -> dict[tuple[int, int], tuple[float, int]]:
    """Correlation and number of shared cells of each pair (side of A, side of B) at one geometry."""
    maps = []
    for ps, v in zip(passes, tex):
        q = footprints(ps, H, z_ref, pitch, dt, yaw)
        maps.append([cell_means(q[:, c], v[:, c], origin, shape) for c in (0, 1)])
    out = {}
    for ca in (0, 1):
        for cb in (0, 1):
            m = np.isfinite(maps[0][ca]) & np.isfinite(maps[1][cb])
            out[ca, cb] = (float(np.corrcoef(maps[0][ca][m], maps[1][cb][m])[0, 1]) if m.sum() > 500 else np.nan, int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path("figs/residual_geometry.png"))
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1],
                    help="sides of A and B whose correlation steers the sweeps (default: A port, B starboard)")
    args = ap.parse_args()
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    full = [load_patch(args.patches, n) for n in "AB"]
    z_ref = float(np.median(np.concatenate([p.poses[:, 2] for p in full])))
    passes = [coarsen_pass(p, args.k) for p in full]
    th = coarsen_theta(theta, args.k)
    r0 = [I.first_echo(p, th) for p in passes]
    tex = [texture(p, r) for p, r in zip(passes, r0)]
    for p, r in zip(passes, r0):
        print(f"pass {p.name}: first echo median port {np.nanmedian(r[:, 0]):.2f} m, stbd {np.nanmedian(r[:, 1]):.2f} m; "
              f"pitch {np.degrees(np.median(p.poses[:, 4])):.1f} deg, sonar {z_ref - np.median(p.poses[:, 2]):+.2f} m below z_ref")
    xy = np.concatenate([p.poses[:, :2] for p in passes])
    origin = xy.min(axis=0) - 55.0
    shape = tuple(int(n) for n in np.ceil((xy.max(axis=0) + 55.0 - origin) / CELL))
    assert len(shape) == 2

    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fig, ax = plt.subplots(1, 4, figsize=(26, 6))
    sweeps = (("level depth H below z_ref (m)", np.arange(12.0, 30.01, 0.5), lambda x: {"H": x}),
              ("pitch offset (deg)", np.arange(-10.0, 10.01, 1.0), lambda x: {"pitch": np.radians(x)}),
              ("ping time shift (s)", np.arange(-3.0, 3.01, 0.25), lambda x: {"dt": x}),
              ("yaw offset (deg)", np.arange(-14.0, 14.01, 1.0), lambda x: {"yaw": np.radians(x)}))
    best = {"H": 21.5, "pitch": 0.0, "dt": 0.0, "yaw": 0.0}
    top = (args.pair[0], args.pair[1])
    for rnd in range(2):  # the second round sweeps each about the first round's peaks
        for a, (label, grid, kw) in zip(ax, sweeps):
            curves = {pr: [] for pr in pairs}
            for x in grid:
                sc = pair_scores(passes, tex, z_ref, origin=origin, shape=shape, **{**best, **kw(x)})
                for pr in pairs:
                    curves[pr].append(sc[pr][0] if sc[pr][1] > 2000 else np.nan)
            x_best = float(grid[int(np.nanargmax(curves[top]))])  # the reference pair sets the next sweeps' centre
            best.update(kw(x_best))
            print(f"round {rnd}, sweep of {label}: " + "; ".join(
                f"A {SIDES[pr[0]]} x B {SIDES[pr[1]]} peak {np.nanmax(curves[pr]):+.3f} at {grid[int(np.nanargmax(curves[pr]))]:g}"
                for pr in pairs if np.isfinite(curves[pr]).any()) + f" -> {x_best:g}")
            if rnd == 1:
                for pr, colr in zip(pairs, COLORS):
                    a.plot(grid, curves[pr], "-o", ms=3, color=colr, label=f"A {SIDES[pr[0]]} x B {SIDES[pr[1]]}")
                a.axhline(0, color="k", lw=0.6)
                a.set(xlabel=label, ylabel="correlation of the passes' textures", title=f"Sweep of the {label.split(' (')[0]}")
                a.legend(frameon=False)
                a.spines[["top", "right"]].set_visible(False)
    print(f"best: H = {best['H']:g} m, pitch offset {np.degrees(best['pitch']):+.1f} deg, time shift {best['dt']:+.2f} s, "
          f"yaw offset {np.degrees(best['yaw']):+.1f} deg")
    for name, kw in (("start as inference has it (nearest first echo; level at the pooled median 18.95 m off track)", {"H": 18.95}),
                     ("level at the sweep's best", best)):
        sc = pair_scores(passes, tex, z_ref, origin=origin, shape=shape, **{"pitch": 0.0, "dt": 0.0, "yaw": 0.0, **kw})
        print(f"  {name}: " + "; ".join(f"A {SIDES[pr[0]]} x B {SIDES[pr[1]]} {sc[pr][0]:+.3f} ({sc[pr][1]} cells)" for pr in pairs))
    # a horizontal offset between the passes (navigation) shows as a peak off zero shift
    fig2, ax2 = plt.subplots(3, 4, figsize=(22, 15))
    zero = {"pitch": 0.0, "dt": 0.0, "yaw": 0.0}
    for row, kw in enumerate(({"H": 18.95, **zero}, {"H": 21.5, **zero}, best)):
        H = f"{kw['H']:g} m, pitch {np.degrees(kw['pitch']):+.0f} deg, {kw['dt']:+.2f} s, yaw {np.degrees(kw['yaw']):+.0f} deg"
        maps = []
        for ps, v in zip(passes, tex):
            q = footprints(ps, kw["H"], z_ref, kw["pitch"], kw["dt"], kw["yaw"])
            maps.append([cell_means(q[:, c], v[:, c], origin, shape) for c in (0, 1)])
        for col, pr in enumerate(pairs):
            a = ax2[row, col]
            surf, ms = shift_search(maps[0][pr[0]], maps[1][pr[1]])
            a.set(title=f"H = {H}: A {SIDES[pr[0]]} x B {SIDES[pr[1]]}", xlabel="shift of B east (m)", ylabel="shift of B north (m)")
            if not np.isfinite(surf).any():
                continue
            pk = np.unravel_index(np.nanargmax(surf), surf.shape)
            sd = 1.4826 * np.nanmedian(np.abs(surf - np.nanmedian(surf)))
            print(f"  H = {H}, A {SIDES[pr[0]]} x B {SIDES[pr[1]]}: at zero shift {surf[ms, ms]:+.3f}; peak {surf[pk]:+.3f} at B shifted "
                  f"({(pk[0] - ms) * CELL:+.1f}, {(pk[1] - ms) * CELL:+.1f}) m east, north; surface sd {sd:.3f}")
            im = a.imshow(surf.T, origin="lower", extent=(-ms * CELL, ms * CELL, -ms * CELL, ms * CELL), cmap="RdBu_r", vmin=-0.3, vmax=0.3)
            fig2.colorbar(im, ax=a, label="correlation")
    fig2.tight_layout()
    fig2.savefig(args.out.with_name(args.out.stem + "_shift.png"), dpi=100)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
