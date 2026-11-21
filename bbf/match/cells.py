r"""Cross-pass check on ground cells: do two passes agree on the echo of the same ground?

A diagnostic of the fit, not a factor. Each bin has a footprint, the echo-weighted mean
location of its part of the seabed curve (`footprint`). Over a square ground cell, the scored
bins of a pass whose footprints fall in it give the sums

.. math::

    Y = \sum_b y_b, \qquad \bar I = \sum_b \bar I_b, \qquad
    k = L\, \frac{(\sum_b \bar I_b)^2}{\sum_b \bar I_b^2},

and :math:`Y` is close to :math:`\mathrm{Gam}(k, \bar I)`: a sum of gamma bins of shape :math:`L`,
matched in mean and variance. For passes :math:`a, b` over the same cell the residual

.. math::

    e = \log\frac{Y_b}{\bar I_b} - \log\frac{Y_a}{\bar I_a}, \qquad
    p(e \mid k_a, k_b) = \frac{t^{k_b} (1 + t)^{-(k_a + k_b)}}{\mathrm{B}(k_a, k_b)}, \quad
    t = \frac{k_b}{k_a} e^{e},

is the log-ratio of two gamma variables of unit mean (:math:`t` is beta prime), whatever the
cell's reflectivity. Speckle averages out over a cell and registration errors below its size
barely matter, so what is left in :math:`e` is what the model gets wrong between the passes:
depths, the angular law, the gain.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import betaln, ndtri_exp
from scipy.stats import beta as beta_dist

from bbf import mesh as M
from bbf.ping.render import sample_weights, split_into_bins
from bbf.ping.theta import Theta


def footprint(mesh: M.Mesh, d, pose, side: int, theta: Theta, u, n_bins: int, dr: float, xi=None):
    r"""Footprint (n_bins, 2) of each bin of one side: the mean of :math:`q_k` under the render's weights; NaN without echo."""
    w, r, q = sample_weights(mesh, d, pose, side, theta, u, dr, xi)
    S = split_into_bins(r, w, n_bins, dr)
    c = jnp.stack([split_into_bins(r, w * q[:, i], n_bins, dr) for i in (0, 1)], axis=-1)
    return jnp.where(S[:, None] > 0, c / jnp.where(S > 0, S, 1.0)[:, None], jnp.nan)


def footprints(mesh: M.Mesh, d, poses, theta: Theta, u, n_bins: int, dr: float, xi=None) -> np.ndarray:
    """Footprints (n, 2, n_bins, 2) of both sides of every pose in poses (n, 6), mesh-local."""
    def both(pose):
        return jnp.stack([footprint(mesh, d, pose, side, theta, u, n_bins, dr, xi) for side in (0, 1)])
    return np.asarray(jax.jit(lambda p: jax.lax.map(both, p))(jnp.asarray(poses)))


@dataclass(frozen=True)
class Cells:
    r"""Sums of one pass over ground cells: cell indices `key` (m, 2), :math:`Y`, :math:`\bar I` and the shape :math:`k` (m,) each."""
    key: np.ndarray
    y: np.ndarray
    mean: np.ndarray
    shape: np.ndarray


def cell_sums(q, y, I_bar, use, cell: float, L: float) -> Cells:
    r"""Sum the bins in `use` over square ground cells of side `cell` (m).

    q: footprints (..., 2); y, I_bar: measured and predicted mean intensities (...); bins
    without a footprint are left out.
    """
    q, y, I_bar = np.asarray(q), np.asarray(y), np.asarray(I_bar)
    ok = np.asarray(use, bool) & np.isfinite(q).all(axis=-1) & (y > 0)
    key, inv = np.unique(np.floor(q[ok] / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    m1, m2 = np.bincount(inv, I_bar[ok], len(key)), np.bincount(inv, I_bar[ok] ** 2, len(key))
    return Cells(key, np.bincount(inv, y[ok], len(key)), m1, L * m1**2 / m2)


def cell_residual(a: Cells, b: Cells) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""(cell indices (m, 2), :math:`e`, :math:`k_a`, :math:`k_b`) over the cells both passes have."""
    lut = {tuple(k): i for i, k in enumerate(b.key.tolist())}
    hit = np.array([lut.get(tuple(k), -1) for k in a.key.tolist()], dtype=int)
    ia, ib = np.flatnonzero(hit >= 0), hit[hit >= 0]
    e = np.log(b.y[ib] / b.mean[ib]) - np.log(a.y[ia] / a.mean[ia])
    return a.key[ia], e, a.shape[ia], b.shape[ib]


def log_ratio_pdf(e, ka, kb):
    r""":math:`\log p(e \mid k_a, k_b)`, the density of :math:`\log(G_b / G_a)` for unit-mean gammas of shapes :math:`k_a, k_b`."""
    lt = np.log(kb / ka) + e
    return kb * lt - (ka + kb) * np.logaddexp(0.0, lt) - betaln(ka, kb)


def ratio_score(e, ka, kb):
    r"""Standard normal score :math:`z = \Phi^{-1}(F(e))` of a residual under :math:`p(e \mid k_a, k_b)`.

    :math:`t / (1 + t) \sim \mathrm{Beta}(k_b, k_a)`. Taken through the log of the nearer tail, so
    far residuals keep their size instead of saturating.
    """
    x = 1.0 / (1.0 + np.exp(-(np.log(kb / ka) + e)))
    lo, hi = beta_dist.logcdf(x, kb, ka), beta_dist.logsf(x, kb, ka)
    return np.where(lo < hi, ndtri_exp(lo), -ndtri_exp(hi))
