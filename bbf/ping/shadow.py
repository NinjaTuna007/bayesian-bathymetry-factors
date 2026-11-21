r"""The shadow factor: lit/shadow labels of range blocks against the predicted visibility.

Each side's bins are grouped into blocks of :math:`B` bins (`Theta.block`). A block is
labelled *lit* when its mean noise-relative intensity exceeds a threshold,

.. math::

    \ell_j = \Bigl[\tfrac{1}{B} \sum_{b \in j} \frac{y_b}{g_c(r_b) N_c} > T\Bigr],

with :math:`T` the :math:`1 - p_\mathrm{fa}` quantile of :math:`\mathrm{Gam}(BL, \text{mean } 1)`,
so a noise-only block is labelled lit with probability :math:`p_\mathrm{fa}` (`Theta.label_fa`).
Labels come from the data and are held fixed while the depths are refined.

The predicted mean of that statistic is :math:`m_j = 1 + \kappa \bar S_j / N_c`, with
:math:`\bar S_j` the block mean of the unit intensity (shadows included through the
visibility). Moment-matching the block mean by :math:`\mathrm{Gam}(BL, \text{mean } m_j)`, the
block passes the threshold with probability :math:`D(m_j) = 1 - P(BL, BL\,T/m_j)`, with :math:`P` the
regularised lower incomplete gamma function. Blocks without a seabed echo are still lit at a
rate :math:`p_0` from ringing, volume scatter and spikes, so (`lit_probability`)

.. math::

    p(\ell_j = \text{lit} \mid d) = p_0 + (1 - p_0) \frac{D(m_j) - p_\mathrm{fa}}{1 - p_\mathrm{fa}}.

:math:`p_0` is best measured from the water column by range (`water_column_lit_rate`); by
default it is :math:`\epsilon_B + (1 - \epsilon_B) p_\mathrm{fa}` with
:math:`\epsilon_B = 1 - (1 - \epsilon_c)^B`. Weak echoes at far range have :math:`m_j` near one,
so a dark block there is little evidence of shadow. Labels in the echo's gradual onset just
before each first echo are left out (`onset_valid`).

Only the dark blocks are scored, by :math:`\log(1 - p)`: the bins of a lit block are scored by
the ping factor's gamma density, so scoring the lit label as well would
count those intensities twice. Each intensity enters one likelihood term.
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
    r"""Lit (True) / shadow (False) label of each block (..., n_blocks) from measured intensities y (..., n_bins)."""
    st = theta.at(side)
    y = np.asarray(y, dtype=float)
    g = np.exp(np.asarray(log_gain(bin_ranges(y.shape[-1], dr), st.gain_db)))
    return np.asarray(_blocks(y / (g * st.noise), theta.block)) > threshold(theta)


def block_mask(labels, n_bins: int, block: int):
    """Per-bin mask (n_bins,) from block labels; bins past the last whole block are False."""
    labels = jnp.asarray(labels, dtype=bool)
    out = jnp.repeat(labels, block)
    return jnp.concatenate([out, jnp.zeros(n_bins - out.shape[0], dtype=bool)])


def noise_lit_rate(theta: Theta, side: int, n_blocks: int) -> np.ndarray:
    r"""Lit rate :math:`p_0` (n_blocks,) of blocks without a seabed echo: measured by range or from the spike rate."""
    if theta.wc_lit is None:
        st = theta.at(side)
        eps_block = 1.0 - (1.0 - st.eps) ** theta.block
        return np.full(n_blocks, eps_block + (1.0 - eps_block) * theta.label_fa)
    rate = np.asarray(theta.wc_lit)[side]
    if len(rate) < n_blocks:
        rate = np.concatenate([rate, np.full(n_blocks - len(rate), rate[-1])])
    return rate[:n_blocks]


def lit_probability(S, log_kappa, theta: Theta, side: int):
    r"""Predicted :math:`p(\ell_j = \text{lit} \mid d)` (n_blocks,) from the unit intensity S (n_bins,).

    With :math:`D(m) = 1 - P(BL, BL\,T/m)` the chance that a block of mean ratio :math:`m` passes the
    threshold (:math:`D(1) = p_\mathrm{fa}`) and :math:`p_0` the lit rate without a seabed echo,

    .. math::

        p = p_0 + (1 - p_0) \frac{D(m) - p_\mathrm{fa}}{1 - p_\mathrm{fa}},

    which is :math:`p_0` where the seabed predicts nothing and tends to 1 for a strong echo.
    """
    k = theta.block * theta.L
    m = 1.0 + jnp.exp(log_kappa) * _blocks(S, theta.block) / theta.at(side).noise
    D = 1.0 - gammainc(k, k * threshold(theta) / m)
    p0 = jnp.asarray(noise_lit_rate(theta, side, m.shape[-1]))
    fa = theta.label_fa
    return p0 + (1.0 - p0) * (D - fa) / (1.0 - fa)


def log_shadow_side(theta: Theta, side: int, labels, S, log_kappa, valid=None):
    r""":math:`\sum_j \log(1 - p_j)` over the valid dark blocks of one side; `log_kappa` is held fixed.

    Lit blocks are left to the ping factor, which scores their intensities.
    """
    p = lit_probability(S, jax.lax.stop_gradient(log_kappa), theta, side)
    p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
    dark = ~jnp.asarray(labels, dtype=bool)
    if valid is not None:
        dark = dark & jnp.asarray(valid, dtype=bool)
    return jnp.sum(jnp.where(dark, jnp.log1p(-p), 0.0))


def onset_valid(r0, n_blocks: int, block_dr: float, onset: float = 1.5) -> np.ndarray:
    """Blocks (n, 2, n_blocks) to use: all but those within `onset` m before each ping-side's first echo r0 (n, 2).

    The echo starts gradually (beam width, off-plane seabed, near-bottom scatter), so labels
    there say little about where the seabed is; NaN r0 leaves every block valid.
    """
    rb = (np.arange(n_blocks) + 0.5) * block_dr
    r0 = np.asarray(r0, dtype=float)[..., None]
    return ~((rb >= r0 - onset) & (rb < r0))


def water_column_lit_rate(labels: list, r0: list, block_dr: float, onset: float = 1.5, smooth: float = 1.0,
                          floor: float = 0.0) -> np.ndarray:
    """Measured lit rate (2, n_blocks) of blocks without a seabed echo, by range.

    Uses the water column of every ping-side: blocks ending more than `onset` m before its
    first echo r0. The rate is smoothed over `smooth` m; beyond the deepest water column
    observed, it continues at the median rate of the outer half of the observed range.
    labels: per pass (n, 2, n_blocks); r0: per pass (n, 2).
    """
    n_blocks = labels[0].shape[-1]
    rb = (np.arange(n_blocks) + 0.5) * block_dr
    out = np.zeros((2, n_blocks))
    w = max(round(smooth / block_dr), 1)
    for c in (0, 1):
        lit = np.zeros(n_blocks)
        cnt = np.zeros(n_blocks)
        for lab, r in zip(labels, r0):
            wc = rb[None, :] < (np.asarray(r)[:, c][:, None] - onset)
            lit += (lab[:, c] & wc).sum(axis=0)
            cnt += wc.sum(axis=0)
        k = np.convolve(lit, np.ones(w), "same")
        n = np.convolve(cnt, np.ones(w), "same")
        seen = n >= 20
        rate = np.where(seen, k / np.maximum(n, 1), np.nan)
        idx = np.flatnonzero(seen)
        if len(idx) == 0:
            out[c] = floor
            continue
        far = rate[idx[len(idx) // 2]:idx[-1] + 1]
        fill = float(np.nanmedian(far))
        rate[idx[-1] + 1:] = fill
        rate[: idx[0]] = rate[idx[0]]
        out[c] = np.clip(np.nan_to_num(rate, nan=fill), floor, 0.95)
    return out


def log_shadow_factor(mesh: M.Mesh, d, pose, labels, theta: Theta, u, dr: float, n_bins: int, log_kappa,
                      valid=None):
    r"""Log shadow factor :math:`\sum_c \sum_{j\,\mathrm{dark}} \log(1 - p_{c,j})` of one ping.

    labels: (2, n_blocks) from `block_labels`; log_kappa: (2,) reflectivity per side, e.g.
    the ping factor's mode. Differentiable in d and the pose.
    """
    total = 0.0
    for side in (0, 1):
        S = Rn.unit_intensity(mesh, d, pose, side, theta, u, n_bins, dr)
        total = total + log_shadow_side(theta, side, labels[side], S, log_kappa[side],
                                        None if valid is None else valid[side])
    return total


def ping_use(labels, n_bins: int, block: int, valid=None):
    """Bins (2, n_bins) the ping factor scores: those in lit blocks.

    labels: (2, n_blocks) from `block_labels`; valid: (2, n_blocks) blocks to use at all (see
    `onset_valid`). Shadow blocks are scored by the shadow factor instead, so each intensity
    enters one likelihood term.
    """
    lit = np.asarray(labels, dtype=bool)
    if valid is not None:
        lit = lit & np.asarray(valid, dtype=bool)
    return np.stack([np.asarray(block_mask(lit[c], n_bins, block)) for c in (0, 1)])
