r"""Starting point of inference: the mesh over the passes and the first echoes.

Each ping's first echo is the nearest seabed in range. Its slant range :math:`r_0`, where the
intensity first jumps far above the water column (`first_echo`), puts a seabed point roughly
below the sonar at
:math:`(x_t, y_t, z_t - r_0)`. A robust plane through those points gives the trend
coefficients :math:`\beta_0` (and the plane depths :math:`X \beta_0`), and the GMRF conditional mean
given the points gives starting depths that follow the relief under the tracks
(`depths_from_points`).
"""
import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cho_factor

from bbf import mesh as M
from bbf.io import Pass
from bbf.match.pairs import swath_box
from bbf.ping.factor import bin_ranges, log_gain
from bbf.ping.theta import Theta
from bbf.prior import Prior


def mesh_for_passes(passes: list[Pass], h: float = 0.5, buffer: float = 10.0) -> M.Mesh:
    r"""Mesh over the union of the passes' swaths plus `buffer`; :math:`z_\mathrm{ref}` is the median sonar height."""
    boxes = np.array([swath_box(p.poses, p.n_bins * p.dr) for p in passes])
    z_ref = float(np.median(np.concatenate([p.poses[:, 2] for p in passes])))
    return M.build(boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(), boxes[:, 3].max(), h=h,
                   buffer=buffer, z_ref=z_ref)


def first_echo(ps: Pass, theta: Theta, min_range: float = 3.0, run: int = 5, jump: float = 20.0,
               window: float = 2.0) -> np.ndarray:
    r"""Slant range (n, 2) of each ping-side's first echo; NaN where none is found.

    The seabed's first echo is a jump of orders of magnitude over the water column, while
    transmit ringing decays smoothly and fish are isolated. With :math:`m_j` the log block mean
    of :math:`y / g N` (blocks of `Theta.block` bins), the jump is found at the first block
    beyond `min_range` metres where the median of the next `run` blocks exceeds the median
    of the preceding `window` metres by the factor `jump`; medians keep one dark speckle
    block or one fish from deciding. The echo is the first block of that run above
    :math:`\sqrt{\text{jump}}`, the half-way level in log, and its range is the block's centre.
    """
    out = np.full((ps.n_pings, 2), np.nan)
    blk = theta.block * ps.dr
    first_block = int(np.ceil(min_range / blk))
    w = max(round(window / blk), 1)
    r = np.asarray(bin_ranges(ps.n_bins, ps.dr))
    for c in (0, 1):
        st = theta.at(c)
        g = np.exp(np.asarray(log_gain(r, st.gain_db)))
        n_blocks = ps.n_bins // theta.block
        z = ps.y[:, c, : n_blocks * theta.block] / (g[: n_blocks * theta.block] * st.noise)
        m = np.log(z.reshape(ps.n_pings, n_blocks, theta.block).mean(axis=2) + 1e-12)
        for t in range(ps.n_pings):
            mt = m[t]
            for j in range(max(first_block, w), n_blocks - run + 1):
                base = np.median(mt[j - w:j])
                if np.median(mt[j:j + run]) > base + np.log(jump):
                    k = j + int(np.argmax(mt[j:j + run] > base + 0.5 * np.log(jump)))
                    out[t, c] = (k + 0.5) * blk
                    break
    return out


def first_echo_points(ps: Pass, r0: np.ndarray) -> np.ndarray:
    """World seabed points (m, 3) below the sonar from the nearer first echo of each ping."""
    r = np.nanmin(np.where(np.isfinite(r0), r0, np.inf), axis=1)
    ok = np.isfinite(r)
    return np.column_stack([ps.poses[ok, 0], ps.poses[ok, 1], ps.poses[ok, 2] - r[ok]])


def plane_fit(points: np.ndarray, mesh: M.Mesh, iters: int = 20, k: float = 1.345) -> np.ndarray:
    r"""Trend coefficients :math:`\beta` of the depth plane :math:`d = \beta_1 x + \beta_2 y + \beta_3` (local).

    Huber IRLS on the depths :math:`z_\mathrm{ref} - z` of world points (m, 3); the scale is the
    MAD of the residuals, so stray echoes (fish, ringing) are down-weighted.
    """
    q = mesh.to_local(points[:, :2])
    A = np.column_stack([q, np.ones(len(q))])
    depth = mesh.z_ref - points[:, 2]
    w = np.ones(len(q))
    beta = np.zeros(3)
    for _ in range(iters):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(A * sw[:, None], depth * sw, rcond=None)[0]
        res = depth - A @ beta
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        u = np.abs(res) / (k * scale)
        w = np.where(u <= 1, 1.0, 1.0 / u)
    return beta


def initial_depths(mesh: M.Mesh, beta: np.ndarray) -> np.ndarray:
    r"""Vertex depths :math:`d_0 = X \beta` of the plane."""
    v = mesh.vertices
    return beta[0] * v[:, 0] + beta[1] * v[:, 1] + beta[2]


def depths_from_points(mesh: M.Mesh, points: np.ndarray, prior: Prior, sigma_obs: float = 0.5) -> np.ndarray:
    r"""Mesh depths from world seabed points: vertex averages, conditional mean of the GMRF elsewhere.

    Each point counts as an observation of its nearest vertex's depth with standard deviation
    `sigma_obs`; the result maximises :math:`\log p(d \mid \beta) + \sum \log \mathcal{N}(\text{obs} \mid d, \sigma_\mathrm{obs}^2)`
    at the prior's trend mean.
    """
    q = mesh.to_local(points[:, :2])
    tri, bary, inside = (np.asarray(a) for a in M.locate(mesh, q))
    vert = mesh.triangles[tri, np.argmax(bary, axis=1)][inside]
    depth = (mesh.z_ref - points[:, 2])[inside]
    n = np.bincount(vert, minlength=mesh.n_vertices).astype(float)
    total = np.bincount(vert, depth, minlength=mesh.n_vertices)
    W = sp.diags(n / sigma_obs**2)
    mean_trend = prior.X @ prior.mu_beta
    rhs = prior.Lambda @ mean_trend + total / sigma_obs**2
    return cho_factor(sp.csc_matrix(prior.Lambda + W)).solve(rhs)
