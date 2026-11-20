r"""The ping factor: bin mixture likelihood with the reflectivity integrated out by Laplace.

For side :math:`c` with unit intensities :math:`S_b` (see `render.unit_intensity`) and
:math:`u = \log \kappa`, the predicted and spike means of bin :math:`b` at range
:math:`r_b = \max(b, \tfrac{1}{2}) \Delta r` are

.. math::

    \bar I_b(u) = g_c(r_b) \bigl(e^u S_b + N_c\bigr), \qquad
    \bar I_{\mathrm{sp},b}(u) = \bar I_b(u) + g_c(r_b) \bigl(\eta_c + \zeta_c e^{-r_b / \ell_c}\bigr),

and each bin is the mixture
:math:`(1 - \epsilon_c) \mathrm{Gam}(y_b \mid L, \bar I_b) + \epsilon_c \mathrm{Gam}(y_b \mid 1, \bar I_{\mathrm{sp},b})`.
With :math:`F(u) = \sum_b \log p(y_b \mid u) + \log \mathcal{N}(u \mid m_c, s_c^2)`, the side's
factor is the Laplace approximation

.. math::

    \log \int e^{F(u)} \mathrm{d}u \approx F(\hat u) + \tfrac{1}{2} \log 2\pi - \tfrac{1}{2} \log H,
    \qquad H = -F''(\hat u).

:math:`\hat u` is found by damped Newton iterations and then refined by one plain Newton
step from its stopped-gradient value. At the mode that step has value :math:`\hat u` and
derivative :math:`-F_{ud} / F_{uu}`, the implicit derivative, so gradients with respect to
the depths and the pose are exact for the approximation.
"""
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from bbf import mesh as M
from bbf.ping import render as Rn
from bbf.ping.theta import Theta

DB_TO_LOG_I = math.log(10) / 10  # gain in dB of amplitude to natural log of intensity
NEWTON_ITERS = 50
MAX_STEP = 1.0  # largest damped Newton step in :math:`u`


def bin_ranges(n_bins: int, dr: float):
    r"""Range :math:`r_b = \max(b, \tfrac{1}{2}) \Delta r` of each bin (bin 0 kept off zero for the log gain)."""
    return jnp.maximum(jnp.arange(n_bins), 0.5) * dr


def log_gain(r, gain_db):
    r""":math:`\log g_c(r)` from `gain_db` :math:`(g_0, g_1, g_2)`: :math:`\tfrac{\ln 10}{10}(g_0 + g_1 r + g_2 \log_{10} r)`."""
    return DB_TO_LOG_I * (gain_db[0] + gain_db[1] * r + gain_db[2] * jnp.log10(r))


def predicted(S, u, theta: Theta, side: int, dr: float):
    r"""Predicted mean :math:`\bar I_b` and spike mean :math:`\bar I_{\mathrm{sp},b}` (n_bins,) at :math:`u = \log\kappa`."""
    st = theta.at(side)
    r = bin_ranges(S.shape[-1], dr)
    g = jnp.exp(log_gain(r, st.gain_db))
    I_bar = g * (jnp.exp(u) * S + st.noise)
    return I_bar, I_bar + g * (st.eta + st.zeta * jnp.exp(-r / st.ell))


def log_gamma(y, k, mean):
    r"""Log density of :math:`\mathrm{Gam}(y \mid k, \mu)`, shape :math:`k` and mean :math:`\mu`."""
    return k * jnp.log(k) - gammaln(k) + (k - 1) * jnp.log(y) - k * jnp.log(mean) - k * y / mean


def log_mixture(y, I_bar, I_sp, L: float, eps: float):
    r"""Per-bin :math:`\log[(1-\epsilon) \mathrm{Gam}(y \mid L, \bar I) + \epsilon \mathrm{Gam}(y \mid 1, \bar I_\mathrm{sp})]`."""
    return jnp.logaddexp(math.log1p(-eps) + log_gamma(y, L, I_bar), math.log(eps) + log_gamma(y, 1.0, I_sp))


def _objective(theta: Theta, side: int, dr: float, y, S, use):
    r""":math:`F(u)` for one side: the bins in `use` plus the Gaussian prior on :math:`u`."""
    st = theta.at(side)
    ok = use & (y > 0)
    y_safe = jnp.where(ok, y, 1.0)

    def F(u):
        I_bar, I_sp = predicted(S, u, theta, side, dr)
        lp = jnp.where(ok, log_mixture(y_safe, I_bar, I_sp, theta.L, st.eps), 0.0)
        z = (u - st.log_kappa_mean) / st.log_kappa_std
        return lp.sum() - 0.5 * z**2 - math.log(st.log_kappa_std) - 0.5 * math.log(2 * math.pi)

    return F, ok, y_safe


def mode(theta: Theta, side: int, dr: float, y, S, use, u0=None, iters: int = NEWTON_ITERS):
    r"""Mode :math:`\hat u` of :math:`F` and the curvature :math:`H = -F''(\hat u)`, differentiable in S.

    u0: a starting value, e.g. the previous evaluation's mode (default: a moment estimate);
    iters: damped Newton iterations, which can be few from a warm start.
    """
    st = theta.at(side)
    F, ok, y_safe = _objective(theta, side, dr, y, S, use)
    dF = jax.grad(F)
    d2F = jax.grad(dF)
    prior_prec = 1.0 / st.log_kappa_std**2

    # start from the moment estimate :math:`\kappa \approx \sum (y/g - N) / \sum S`
    r = bin_ranges(S.shape[-1], dr)
    g = jnp.exp(log_gain(r, st.gain_db))
    num = jnp.sum(jnp.where(ok, jnp.maximum(y_safe / g - st.noise, 0.0), 0.0))
    den = jnp.sum(jnp.where(ok, S, 0.0))
    good = (num > 0) & (den > 0)
    if u0 is None:
        u0 = jnp.where(good, jnp.log(jnp.where(good, num / jnp.where(den > 0, den, 1.0), 1.0)), st.log_kappa_mean)

    def step(_, u):
        h = jnp.maximum(-d2F(u), prior_prec)  # keep the step uphill where F is not concave
        return u + jnp.clip(dF(u) / h, -MAX_STEP, MAX_STEP)

    u_star = jax.lax.stop_gradient(jax.lax.fori_loop(0, iters, step, jax.lax.stop_gradient(u0)))
    H_star = jnp.maximum(-d2F(u_star), 1e-12)
    # one plain Newton step for the implicit derivative of the mode; tiny at a converged mode, and
    # limited like the damped steps where F is flat in u (e.g. no echo predicted)
    u_hat = u_star + jnp.clip(dF(u_star) / H_star, -MAX_STEP, MAX_STEP)
    return u_hat, jnp.maximum(-d2F(u_hat), 1e-12)


def log_ping_side(theta: Theta, side: int, dr: float, y, S, use=None):
    r"""Laplace approximation of :math:`\log \int \prod_b p(y_b \mid S, u)\, p(u)\, \mathrm{d}u` for one side.

    y, S: (n_bins,) measured intensity and unit intensity; use: bins to include (default
    all), e.g. without the matched bins. Bins with :math:`y \le 0` are always left out.
    """
    use = jnp.ones(S.shape, dtype=bool) if use is None else jnp.asarray(use, dtype=bool)
    F, _, _ = _objective(theta, side, dr, y, S, use)
    u_hat, H = mode(theta, side, dr, y, S, use)
    return F(u_hat) + 0.5 * math.log(2 * math.pi) - 0.5 * jnp.log(H)


def log_ping_factor(mesh: M.Mesh, d, pose, y, theta: Theta, u, dr: float, use=None):
    r"""Log ping factor :math:`\log p(y_t \mid d, x_t)`, summed over port and starboard.

    y: (2, n_bins) measured intensities (amplitude squared); u: horizontal curve samples
    (K,); use: (2, n_bins) bins to include (default all). Differentiable in d and the pose.
    """
    y = jnp.asarray(y)
    n_bins = y.shape[-1]
    total = 0.0
    for side in (0, 1):
        S = Rn.unit_intensity(mesh, d, pose, side, theta, u, n_bins, dr)
        total = total + log_ping_side(theta, side, dr, y[side], S, None if use is None else use[side])
    return total
