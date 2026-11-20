r"""The shadow factor: lit/shadow labels of range blocks against the predicted visibility.

Each side's bins are grouped into blocks of :math:`B` bins (`Theta.block`). A block is
labelled *lit* when its mean noise-relative intensity exceeds a threshold,

.. math::

    \ell_j = \Bigl[\tfrac{1}{B} \sum_{b \in j} \frac{y_b}{g_c(r_b) N_c} > T\Bigr],

with :math:`T` the :math:`1 - p_\mathrm{fa}` quantile of :math:`\mathrm{Gam}(BL, \text{mean } 1)`,
so a noise-only block is labelled lit with probability :math:`p_\mathrm{fa}` (`Theta.label_fa`).
Labels come from the data and are held fixed, like the matched pairs.

The predicted mean of that statistic is :math:`m_j = 1 + \kappa \bar S_j / N_c`, with
:math:`\bar S_j` the block mean of the unit intensity (shadows included through the soft
visibility). Moment-matching the block mean by :math:`\mathrm{Gam}(BL, \text{mean } m_j)` and
allowing a spike in the block with probability :math:`\epsilon_B = 1 - (1 - \epsilon_c)^B`,

.. math::

    p(\ell_j = \text{lit} \mid d) = \epsilon_B + (1 - \epsilon_B)
        \Bigl(1 - P\bigl(BL,\; BL\, T / m_j\bigr)\Bigr),

with :math:`P` the regularised lower incomplete gamma function. Shadow and water column give
:math:`m_j = 1`, so the lit probability falls to the false-alarm level; weak echoes at far
range have :math:`m_j` near one too, so a dark block there is little evidence of shadow.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammainc
from scipy.special import gammaincinv

from bbf import mesh as M
from bbf.ping import render as Rn
from bbf.ping.factor import bin_ranges, log_gain
from bbf.ping.theta import Theta


def threshold(theta: Theta) -> float:
    r"""Label threshold :math:`T` on the block-mean noise-relative intensity."""
    k = theta.block * theta.L
    return float(gammaincinv(k, 1.0 - theta.label_fa)) / k


def _blocks(x, block: int):
    """Block means of the last axis, dropping a partial last block."""
    n = x.shape[-1] // block
    return x[..., : n * block].reshape(*x.shape[:-1], n, block).mean(axis=-1)


def block_labels(y, theta: Theta, side: int, dr: float) -> np.ndarray:
    r"""Lit (True) / shadow (False) label of each block (n_blocks,) from measured intensities y (n_bins,)."""
    st = theta.at(side)
    y = np.asarray(y, dtype=float)
    g = np.exp(np.asarray(log_gain(bin_ranges(y.shape[-1], dr), st.gain_db)))
    return np.asarray(_blocks(y / (g * st.noise), theta.block)) > threshold(theta)


def block_mask(labels, n_bins: int, block: int):
    """Per-bin mask (n_bins,) from block labels; bins past the last whole block are False."""
    labels = jnp.asarray(labels, dtype=bool)
    out = jnp.repeat(labels, block)
    return jnp.concatenate([out, jnp.zeros(n_bins - out.shape[0], dtype=bool)])


def lit_probability(S, log_kappa, theta: Theta, side: int):
    r"""Predicted :math:`p(\ell_j = \text{lit} \mid d)` (n_blocks,) from the unit intensity S (n_bins,)."""
    st = theta.at(side)
    k = theta.block * theta.L
    m = 1.0 + jnp.exp(log_kappa) * _blocks(S, theta.block) / st.noise
    eps_block = 1.0 - (1.0 - st.eps) ** theta.block
    return eps_block + (1.0 - eps_block) * (1.0 - gammainc(k, k * threshold(theta) / m))


def log_shadow_side(theta: Theta, side: int, labels, S, log_kappa):
    r""":math:`\sum_j \log p(\ell_j \mid d)` for one side; `log_kappa` is held fixed."""
    p = lit_probability(S, jax.lax.stop_gradient(log_kappa), theta, side)
    p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
    labels = jnp.asarray(labels, dtype=bool)
    return jnp.sum(jnp.where(labels, jnp.log(p), jnp.log1p(-p)))


def log_shadow_factor(mesh: M.Mesh, d, pose, labels, theta: Theta, u, dr: float, n_bins: int, log_kappa):
    r"""Log shadow factor :math:`\sum_c \sum_j \log p(\ell_{c,j} \mid d)` of one ping.

    labels: (2, n_blocks) from `block_labels`; log_kappa: (2,) reflectivity per side, e.g.
    the ping factor's mode. Differentiable in d and the pose.
    """
    total = 0.0
    for side in (0, 1):
        S = Rn.unit_intensity(mesh, d, pose, side, theta, u, n_bins, dr)
        total = total + log_shadow_side(theta, side, labels[side], S, log_kappa[side])
    return total


def ping_use(labels, n_bins: int, block: int, matched=None):
    """Bins (2, n_bins) the ping factor scores: those in lit blocks, minus matched bins.

    labels: (2, n_blocks) from `block_labels`; matched: (2, n_bins) bins taken by the match
    factor (default none). Shadow blocks are scored by the shadow factor instead, and
    matched bins by the match factor, so each intensity enters one likelihood term.
    """
    use = np.stack([np.asarray(block_mask(labels[c], n_bins, block)) for c in (0, 1)])
    return use if matched is None else use & ~np.asarray(matched, dtype=bool)
