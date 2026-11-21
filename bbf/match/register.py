r"""Horizontal offset between two passes from their textures, before any model of the echo.

The match factor ties the passes through one reflectivity field, so their footprints must agree
to well within the field's range :math:`\rho_\xi`. Navigation between passes minutes apart need
not: an offset of a few metres leaves the shared texture uncorrelated and the field is then
fitted to each pass's texture in turn. This module estimates the offset so that it can be
taken out of the poses (`bbf.io.Pass.shifted`) or start an unknown offset per pass.

Every bin beyond its ping-side's first echo is put on a level seabed at a given height: in the
ping plane (normal :math:`R e_x`, as `bbf.ping.geometry.trace`) at slant range :math:`r`. Its value
is the log intensity minus a running mean along range, which removes gain, beam and angular
response and keeps the texture. The values are averaged over square ground cells per pass and
side, and the two maps of a pair of sides are correlated over the cells both cover, for every
move of the second pass by whole cells. Seabed relief shades with opposite sign from opposite
directions, so only pairs of sides that look the same way are used. The peak is the move to
give the second pass.

A level seabed is the crudest surface. A depth error :math:`\delta` moves the footprints of the two
passes apart by :math:`\delta\, h (1 / x_a + 1 / x_b)` across track (altitude :math:`h`, ground ranges
:math:`x`), which varies over the swath and only blurs the peak; an offset of the navigation, a
shift of the ping times or a pitch offset moves all of it together.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from bbf.io import Pass

PORT, STARBOARD = 0, 1


def texture(ps: Pass, r0, window: float = 3.0, skip: float = 1.0) -> np.ndarray:
    r"""Log intensity minus its running mean over `window` m of range, (n, 2, B).

    NaN up to `skip` m beyond each ping-side's first echo r0 (n, 2), where r0 is NaN, and where
    :math:`y \le 0`.
    """
    r = np.arange(ps.n_bins) * ps.dr
    start = np.where(np.isfinite(r0), r0, np.inf)[:, :, None] + skip
    ok = (ps.y > 0) & (r[None, None] > start)
    ly = np.where(ok, np.log(np.where(ok, ps.y, 1.0)), 0.0)
    w = max(round(window / ps.dr), 3)
    cnt = uniform_filter1d(ok.astype(float), w, axis=-1, mode="constant")
    mean = uniform_filter1d(ly, w, axis=-1, mode="constant") / np.maximum(cnt, 1e-9)
    return np.where(ok & (cnt > 0.5), ly - mean, np.nan)


def level_footprints(ps: Pass, z_seabed: float) -> np.ndarray:
    r"""Ground points (n, 2, B, 2) of every bin on a level seabed at world height `z_seabed`; NaN where the range does not reach it."""
    poses = ps.poses
    r = np.arange(ps.n_bins) * ps.dr
    yaw, pitch = poses[:, 5], poses[:, 4]
    # heading :math:`R e_x` in east, north, up (`bbf.ping.geometry.pose_frame`); the roll does not enter
    hd = np.column_stack([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    fwd = hd[:, :2] / np.linalg.norm(hd[:, :2], axis=1, keepdims=True)
    below = poses[:, 2] - z_seabed
    out = np.full((ps.n_pings, 2, ps.n_bins, 2), np.nan)
    for c, sign in ((PORT, 1.0), (STARBOARD, -1.0)):
        o = sign * np.column_stack([-fwd[:, 1], fwd[:, 0], np.zeros(len(fwd))])  # horizontal outboard axis, in the ping plane
        w = np.cross(o, hd)
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        w = np.where(w[:, 2:3] > 0, -w, w)  # the plane's other axis, pointing down
        with np.errstate(divide="ignore", invalid="ignore"):
            cg = below[:, None] / (r[None] * np.abs(w[:, 2:3]))
        ok = cg <= 1.0
        sg = np.sqrt(np.where(ok, 1.0 - cg**2, np.nan))
        cg = np.where(ok, cg, np.nan)
        out[:, c] = poses[:, None, :2] + r[None, :, None] * (cg[..., None] * w[:, None, :2] + sg[..., None] * o[:, None, :2])
    return out


def cell_means(q, v, origin, shape: tuple[int, int], cell: float, min_count: int = 3) -> np.ndarray:
    """Mean of values v (...) at points q (..., 2) over square cells of side `cell`, (nx, ny); NaN below `min_count` values."""
    ok = np.isfinite(v) & np.isfinite(q).all(axis=-1)
    ij = np.floor((q[ok] - np.asarray(origin)) / cell).astype(int)
    inb = (ij >= 0).all(axis=1) & (ij < np.array(shape)).all(axis=1)
    flat = ij[inb, 0] * shape[1] + ij[inb, 1]
    cnt = np.bincount(flat, minlength=shape[0] * shape[1])
    tot = np.bincount(flat, v[ok][inb], minlength=shape[0] * shape[1])
    return np.where(cnt >= min_count, tot / np.maximum(cnt, 1), np.nan).reshape(shape)


def move_surface(a: np.ndarray, b: np.ndarray, max_cells: int, min_cells: int = 1000) -> np.ndarray:
    """Correlation of map a with map b moved by (i, j) cells, for i, j in -max_cells..max_cells; NaN under `min_cells` shared cells.

    Entry [i + max_cells, j + max_cells] has b's content moved by (i, j) cells, i.e. a(x) against b(x - move).
    """
    n = 2 * max_cells + 1
    out = np.full((n, n), np.nan)
    pad = np.full((b.shape[0] + 2 * max_cells, b.shape[1] + 2 * max_cells), np.nan)
    pad[max_cells:max_cells + b.shape[0], max_cells:max_cells + b.shape[1]] = b
    fa = np.isfinite(a)
    for i in range(n):
        for j in range(n):
            bs = pad[2 * max_cells - i:2 * max_cells - i + a.shape[0], 2 * max_cells - j:2 * max_cells - j + a.shape[1]]
            m = fa & np.isfinite(bs)
            if m.sum() >= min_cells:
                out[i, j] = np.corrcoef(a[m], bs[m])[0, 1]
    return out


@dataclass(frozen=True)
class Offset:
    r"""The move (east, north; m) to give the second pass, with the evidence for it.

    corr, corr_zero: the textures' correlation at the move and with the pass left where it is;
    z: the peak's height over the surface's median in robust standard deviations of the surface
    (a few is noise; ten is a clear peak); sides: the pair of sides (of the first, of the second
    pass) it came from; surface, moves: the correlation over the grid of moves (n, n) and the
    grid's values along either axis (n,).
    """
    move: np.ndarray
    corr: float
    corr_zero: float
    z: float
    sides: tuple[int, int]
    surface: np.ndarray
    moves: np.ndarray


def same_look_pairs(a: Pass, b: Pass) -> list[tuple[int, int]]:
    """Pairs (side of a, side of b) that look the same way: unlike sides for opposite headings, like sides otherwise."""
    ha, hb = (np.array([np.sin(p.poses[:, 5]).mean(), np.cos(p.poses[:, 5]).mean()]) for p in (a, b))
    return [(PORT, STARBOARD), (STARBOARD, PORT)] if ha @ hb < 0 else [(PORT, PORT), (STARBOARD, STARBOARD)]


def offset_between(a: Pass, b: Pass, r0_a, r0_b, z_seabed: float, cell: float = 0.5, max_move: float = 6.0,
                   window: float = 3.0, reach: float = 55.0, min_cells: int = 1000) -> Offset | None:
    r"""The horizontal move of pass b that best lines its texture up with pass a's, on a level seabed at `z_seabed`.

    r0_a, r0_b: first echoes (n, 2) of the passes (`bbf.inference.init.first_echo`); cell: side of
    the ground cells (m), also the grid of moves, out to `max_move`; reach: how far beyond the
    tracks the cell maps extend (m). The pair of same-looking sides with the clearest peak
    decides, and the peak is refined within its cell by a parabola through its neighbours along
    each axis. None when no pair shares `min_cells` cells.
    """
    xy = np.concatenate([a.poses[:, :2], b.poses[:, :2]])
    origin = xy.min(axis=0) - reach
    nx, ny = (int(v) for v in np.ceil((xy.max(axis=0) + reach - origin) / cell))
    shape = (nx, ny)
    qa, qb = level_footprints(a, z_seabed), level_footprints(b, z_seabed)
    ta, tb = texture(a, r0_a, window), texture(b, r0_b, window)
    mc = round(max_move / cell)
    moves = np.arange(-mc, mc + 1) * cell
    best: Offset | None = None
    for ca, cb in same_look_pairs(a, b):
        surf = move_surface(cell_means(qa[:, ca], ta[:, ca], origin, shape, cell),
                            cell_means(qb[:, cb], tb[:, cb], origin, shape, cell), mc, min_cells)
        if not np.isfinite(surf).any():
            continue
        i, j = (int(v) for v in np.unravel_index(np.nanargmax(surf), surf.shape))
        med = np.nanmedian(surf)
        z = float((surf[i, j] - med) / max(1.4826 * np.nanmedian(np.abs(surf - med)), 1e-12))
        move = np.array([moves[i], moves[j]])
        for ax, k in enumerate((i, j)):
            if 0 < k < len(moves) - 1:
                c = [surf[k + s, j] if ax == 0 else surf[i, k + s] for s in (-1, 0, 1)]
                den = c[0] - 2 * c[1] + c[2]
                if np.isfinite(den) and den < 0:
                    move[ax] += 0.5 * (c[0] - c[2]) / den * cell
        found = Offset(move=move, corr=float(surf[i, j]), corr_zero=float(surf[mc, mc]), z=z, sides=(ca, cb),
                       surface=surf, moves=moves)
        if best is None or found.z > best.z:
            best = found
    return best


def align_passes(passes: list[Pass], r0: list, z_seabed: float, min_z: float = 5.0,
                 **kwargs) -> tuple[list[Pass], list[Offset | None]]:
    r"""The passes with each one after the first moved onto the first's texture, and what was found per pass.

    Each later pass is moved by `offset_between` against the first (kwargs go to it) when the peak
    stands at least `min_z` robust standard deviations over its surface; otherwise, and where the
    two share too few cells, it stays where the navigation has it. The entry of the first pass is
    None. The moves are estimates to start from: an offset per pass is properly an unknown of the
    graph.
    """
    out: list[Pass] = [passes[0]]
    found: list[Offset | None] = [None]
    for ps, r in zip(passes[1:], r0[1:]):
        off = offset_between(passes[0], ps, r0[0], r, z_seabed, **kwargs)
        found.append(off)
        out.append(ps.shifted(*off.move) if off is not None and off.z >= min_z else ps)
    return out, found
