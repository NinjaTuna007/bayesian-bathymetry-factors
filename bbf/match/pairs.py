r"""Matched pairs between two passes: chosen in the outer loop, held fixed while d is refined.

Locations :math:`q` are drawn uniformly (`draw_overlap`) or on a dense grid (`grid_overlap`)
where the two passes' swaths overlap. At the current
depths each pass sends :math:`q` to the ping whose plane passes nearest the seabed point
:math:`P(q)`, the side it lies on and the bin of its slant range. A draw becomes a pair
when both passes see it, the echo dominates the background (below), and neither of its
bins is already in a pair, so every bin enters at most one pair.

Echo guard (paper, eq. pair selection): with :math:`\bar J = g S_b` the bin's predicted
intensity per unit reflectivity, keep the pair when

.. math::

    \frac{y}{\bar J} + \frac{y'}{\bar J'} > 2\,\text{margin}\,
        \max\Bigl(\frac{g N}{\bar J}, \frac{g' N'}{\bar J'}\Bigr).

The left side estimates :math:`2\kappa_q` independently of the pair's ratio, and the right
side depends on the depths only, so selecting on it leaves the density of the residual
unchanged. Shadowed bins have tiny :math:`S_b` and fail the guard.
"""
from dataclasses import dataclass

import jax
import numpy as np

from bbf import mesh as M
from bbf.ping import geometry as G
from bbf.ping.factor import bin_ranges, log_gain
from bbf.ping.theta import Theta


@dataclass(frozen=True)
class Pairs:
    """Matched pairs: ping, side and bin of each pair's bin in pass a and in pass b, (m,) each."""
    ping_a: np.ndarray
    side_a: np.ndarray
    bin_a: np.ndarray
    ping_b: np.ndarray
    side_b: np.ndarray
    bin_b: np.ndarray

    def __len__(self) -> int:
        return len(self.ping_a)

    @property
    def sign(self) -> np.ndarray:
        r"""Side offset sign :math:`\sigma_m = c'_m - c_m`: :math:`+1` from port to starboard, :math:`-1` the reverse, 0 on the same side."""
        return self.side_b - self.side_a


def swath_box(poses, max_range: float) -> tuple[float, float, float, float]:
    """Axis-aligned box (xmin, xmax, ymin, ymax) holding every ping's swath, from the poses (n, 6)."""
    poses = np.asarray(poses)
    yaw = poses[:, 5]
    sx, sy = max_range * np.cos(yaw), -max_range * np.sin(yaw)  # starboard is :math:`(\cos\psi, -\sin\psi)`
    ex = np.concatenate([poses[:, 0] + sx, poses[:, 0] - sx])
    ey = np.concatenate([poses[:, 1] + sy, poses[:, 1] - sy])
    return float(ex.min()), float(ex.max()), float(ey.min()), float(ey.max())


def _overlap_box(poses_a, poses_b, max_range: float):
    a, b = swath_box(poses_a, max_range), swath_box(poses_b, max_range)
    return max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])


def draw_overlap(poses_a, poses_b, max_range: float, n: int, rng=None) -> np.ndarray:
    """n locations (n, 2) uniform in the intersection of the two passes' swath boxes; (0, 2) if disjoint."""
    x0, x1, y0, y1 = _overlap_box(poses_a, poses_b, max_range)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 2))
    return np.random.default_rng(rng).uniform([x0, y0], [x1, y1], size=(n, 2))


def grid_overlap(poses_a, poses_b, max_range: float, spacing: float, rng=None) -> np.ndarray:
    """Locations (m, 2) on a square grid of `spacing` over the intersection of the swath boxes, randomly offset.

    Dense, so most bins both passes see get a candidate pair; the random offset changes
    which bins are tried from round to round.
    """
    x0, x1, y0, y1 = _overlap_box(poses_a, poses_b, max_range)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 2))
    off = np.random.default_rng(rng).uniform(0, spacing, 2)
    gx, gy = np.meshgrid(np.arange(x0 + off[0], x1, spacing), np.arange(y0 + off[1], y1, spacing))
    return np.column_stack([gx.ravel(), gy.ravel()])


def project(mesh: M.Mesh, d, poses, q, n_bins: int, dr: float, chunk: int = 4096):
    r"""Ping, side and bin (m,) each that a pass sees seabed point :math:`P(q)` in; :math:`-1` where unseen.

    The ping is the one whose plane passes nearest :math:`P(q)`, among pings within the
    maximum range horizontally, and only if the planes pass on both sides of it (so a
    point beyond the pass's ends is not snapped to its first or last ping). The side is
    the sign of the sonar-frame port coordinate; the bin is the nearest to the slant range.
    """
    q = np.asarray(q)
    if len(q) > chunk:  # the ping-plane distances are (points, pings): keep them small
        parts = [project(mesh, d, poses, q[i:i + chunk], n_bins, dr, chunk) for i in range(0, len(q), chunk)]
        return tuple(np.concatenate(x) for x in zip(*parts, strict=True))
    poses = np.asarray(poses)
    P = np.asarray(M.seabed_point(mesh, d, q))
    s, R = (np.asarray(a) for a in jax.vmap(G.pose_frame)(poses))
    hd = R[:, :, 0]
    dist = P @ hd.T - np.einsum("ij,ij->i", s, hd)  # (m, n) signed distance to each ping plane
    horiz = np.hypot(P[:, None, 0] - s[None, :, 0], P[:, None, 1] - s[None, :, 1])
    dist = np.where((horiz <= n_bins * dr) & np.isfinite(P[:, None, 2]), dist, np.nan)
    ok = np.isfinite(dist).any(axis=1)
    with np.errstate(invalid="ignore"):
        ok[ok] = (np.nanmin(dist[ok], axis=1) <= 0) & (np.nanmax(dist[ok], axis=1) >= 0)
    ping = np.full(len(P), -1)
    ping[ok] = np.nanargmin(np.abs(dist[ok]), axis=1)
    side, bins = np.full(len(P), -1), np.full(len(P), -1)
    k = np.flatnonzero(ok)
    v = P[k] - s[ping[k]]
    port = np.einsum("mji,mj->mi", R[ping[k]], v)[:, 1] >= 0
    b = np.rint(np.linalg.norm(v, axis=1) / dr).astype(int)
    inr = b < n_bins
    side[k[inr]] = np.where(port[inr], 0, 1)
    bins[k[inr]] = b[inr]
    ping[k[~inr]] = -1
    return ping, side, bins


def select_pairs(proj_a, proj_b, S_a, S_b, y_a, y_b, theta: Theta, dr: float, margin: float = 10.0,
                 taken_a=None, taken_b=None) -> Pairs:
    """Matched pairs from the two passes' projections of the same draws.

    proj_a, proj_b: (ping, side, bin) from `project`; S_a, S_b: (n, 2, B) unit intensities
    of every ping of each pass at the current depths; y_a, y_b: (n, 2, B) intensities.
    taken_a, taken_b: (n, 2, B) bins already in a pair with a third pass, left out so that
    each bin is in at most one pair.
    """
    pa, ca, ba = (np.asarray(x) for x in proj_a)
    pb, cb, bb = (np.asarray(x) for x in proj_b)
    seen = (pa >= 0) & (pb >= 0) & (ca >= 0) & (cb >= 0)
    k = np.flatnonzero(seen)
    pa, ca, ba, pb, cb, bb = pa[k], ca[k], ba[k], pb[k], cb[k], bb[k]
    Sa, Sb = np.asarray(S_a)[pa, ca, ba], np.asarray(S_b)[pb, cb, bb]
    ya, yb = np.asarray(y_a)[pa, ca, ba], np.asarray(y_b)[pb, cb, bb]
    r = np.asarray(bin_ranges(np.asarray(y_a).shape[-1], dr))
    ga, gb = _gain(theta, ca, r[ba]), _gain(theta, cb, r[bb])
    Na, Nb = np.asarray(theta.noise)[ca], np.asarray(theta.noise)[cb]
    usable = (Sa > 0) & (Sb > 0) & (ya > 0) & (yb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ya / (ga * Sa) + yb / (gb * Sb)
        thresh = 2 * margin * np.maximum(Na / Sa, Nb / Sb)
    keep = usable & (stat > thresh)
    if taken_a is not None:
        keep &= ~np.asarray(taken_a)[pa, ca, ba]
    if taken_b is not None:
        keep &= ~np.asarray(taken_b)[pb, cb, bb]
    idx = np.flatnonzero(keep)
    # one pair per bin: the first draw that uses a bin of pass a, then of pass b, wins
    for p_, c_, b_ in ((pa, ca, ba), (pb, cb, bb)):
        key = np.stack([p_[idx], c_[idx], b_[idx]], axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        idx = idx[np.sort(first)]
    return Pairs(pa[idx], ca[idx], ba[idx], pb[idx], cb[idx], bb[idx])


def _gain(theta: Theta, side, r):
    """Receiver gain :math:`g_c(r)` for each (side, range) pair."""
    return np.exp(np.asarray(log_gain(r, np.asarray(theta.gain_db)[side])))


def matched_mask(pairs: Pairs, n_pings: int, n_bins: int, which: str) -> np.ndarray:
    """(n_pings, 2, n_bins) mask of the bins of pass `which` ("a" or "b") that are in a pair.

    The pass's ping factor leaves them out (`use=~mask`), so each bin enters one factor.
    """
    p, c, b = (pairs.ping_a, pairs.side_a, pairs.bin_a) if which == "a" else (pairs.ping_b, pairs.side_b, pairs.bin_b)
    out = np.zeros((n_pings, 2, n_bins), bool)
    out[p, c, b] = True
    return out
