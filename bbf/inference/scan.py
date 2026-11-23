r"""Per-ping-side planes fitted to the intensities: a diagnostic, falsified as a start.

On real data the planes of two passes disagreed by metres where they see the same ground:
the slopes absorbed errors in the gain and beam pattern (`scripts/scan_overlap.py`). The scan
is kept for that diagnostic; inference starts from the first echoes (`bbf.inference.init`).

A single plane through the first echoes cannot tell relief under the tracks from a tilt: a
bump below the passes raises their first echoes, and the plane tilts to match. Each side of
each ping is fitted with its own plane instead,

.. math::

    z(u) = s_z - D + \tan\gamma \; u,

with :math:`u` the horizontal distance outboard, :math:`D` the depth below the sonar at nadir and
:math:`\tan\gamma` the outboard slope (positive rising). Candidates are scored by the ping and shadow
factors (`bbf.ping.factor`, `bbf.ping.shadow`) of that ping-side alone, coarse to fine as in the
prototype's ``geometry_scan``: depth at zero slope, slope at that depth, alternating local depth
and slope scans (the two trade off along a ridge), then both jointly.
A plane needs no mesh, so each candidate is rendered in closed form.
"""
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from bbf.io import Pass
from bbf.ping import geometry as G
from bbf.ping.factor import laplace_ping_side
from bbf.ping.render import curve_length, split_into_bins
from bbf.ping.shadow import log_shadow_side, ping_use
from bbf.ping.theta import Theta


def plane_samples(pose, side: int, D, slope, u):
    r"""Seabed points (K, 3) and upward normals (K, 3) of the plane :math:`z = s_z - D + \text{slope}\,u` in the ping plane."""
    s, hd, nxy, fwd, out = G.side_frame(pose, side)
    u = jnp.asarray(u)
    z = s[2] - D + slope * u
    w = -hd[2] * (z - s[2]) / nxy  # along-track shift that keeps the point in the ping plane
    q = s[:2] + u[:, None] * out + w[:, None] * fwd
    P = jnp.concatenate([q, z[:, None]], axis=-1)
    n = jnp.concatenate([-slope * out, jnp.ones(1)]) / jnp.sqrt(1 + slope**2)
    return P, jnp.broadcast_to(n, P.shape)


def plane_unit_intensity(pose, side: int, theta: Theta, D, slope, u, n_bins: int, dr: float):
    r"""Unit intensity (n_bins,) of one side over a plane seabed; a plane seen from above hides nothing."""
    P, n = plane_samples(pose, side, D, slope, u)
    a, r = G.echo(theta, pose, side, P, n)
    w = jnp.where(P[:, 2] <= theta.z_water, a * curve_length(P[:, :2]), 0.0) / dr
    return split_into_bins(r, w, n_bins, dr)


SCAN_ITERS = 15  # damped Newton iterations of each candidate's log-kappa mode; 8-50 pick the same planes


@cache
def _scorer(theta: Theta, side: int, dr: float, n_bins: int, iters: int = SCAN_ITERS):
    """Jitted score of many (D, slope) candidates for one ping-side: ping plus shadow factor."""
    def one(pose, y, use, labels, valid, u, D, slope):
        S = plane_unit_intensity(pose, side, theta, D, slope, u, n_bins, dr)
        ping, u_hat = laplace_ping_side(theta, side, dr, y, S, use, iters=iters)
        return ping + log_shadow_side(theta, side, labels, S, u_hat, valid, dr)
    return jax.jit(jax.vmap(one, in_axes=(None, None, None, None, None, None, 0, 0)))


def scan_ping_side(pose, side: int, theta: Theta, y, labels, valid, u, n_bins: int, dr: float, r0: float,
                   iters: int = SCAN_ITERS):
    r"""Best (D, slope) of one ping-side, coarse to fine; r0 is its first echo (NaN: unknown)."""
    use = ping_use(np.stack([labels, labels]), n_bins, theta.block, valid=np.stack([valid, valid]))[0]
    f = _scorer(theta, side, dr, n_bins, iters)
    args = (jnp.asarray(pose), jnp.asarray(y), jnp.asarray(use), jnp.asarray(labels), jnp.asarray(valid), jnp.asarray(u))

    def best(D, sl):
        sc = np.asarray(f(*args, jnp.asarray(D, float), jnp.asarray(sl, float)))
        k = int(np.nanargmax(sc))
        return float(D[k]), float(sl[k]), float(sc[k])

    lo = (r0 - 2.0) if np.isfinite(r0) else 3.0
    hi = (r0 + 8.0) if np.isfinite(r0) else n_bins * dr * 0.8
    Ds = np.arange(max(lo, 1.0), hi, 0.25)
    D, _, _ = best(Ds, np.zeros_like(Ds))
    sl = np.arange(-1.0, 1.0001, 0.05)
    _, slope, _ = best(np.full_like(sl, D), sl)
    # depth and slope trade off along a ridge: alternate local scans before the joint one
    for _ in range(3):
        Ds = D + np.arange(-2.0, 2.0001, 0.1)
        D, _, _ = best(Ds, np.full_like(Ds, slope))
        sl = slope + np.arange(-0.3, 0.3001, 0.02)
        _, slope, _ = best(np.full_like(sl, D), sl)
    DD, SS = np.meshgrid(D + np.arange(-0.5, 0.501, 0.1), slope + np.arange(-0.1, 0.101, 0.02), indexing="ij")
    return best(DD.ravel(), SS.ravel())


def scan_pass(ps: Pass, theta: Theta, labels, valid, r0, u, stride: int = 10, log=None) -> np.ndarray:
    """(D, slope, score) of every `stride`-th ping (m, 2, 3); NaN where a side has no usable data."""
    idx = np.arange(0, ps.n_pings, stride)
    out = np.full((len(idx), 2, 3), np.nan)
    for i, t in enumerate(idx):
        for c in (0, 1):
            if not labels[t, c].any():
                continue
            out[i, c] = scan_ping_side(ps.poses[t], c, theta, ps.y[t, c], labels[t, c], valid[t, c], u, ps.n_bins,
                                       ps.dr, r0[t, c])
        if log is not None and i % 20 == 0:
            log(f"  scan {ps.name}: ping {t}: port D {out[i, 0, 0]:.2f} slope {out[i, 0, 1]:+.2f}, "
                f"stbd D {out[i, 1, 0]:.2f} slope {out[i, 1, 1]:+.2f}")
    return out
