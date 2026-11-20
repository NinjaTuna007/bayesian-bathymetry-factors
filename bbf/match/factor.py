r"""The match factor: matched pairs of bins with a shared reflectivity integrated out.

For a pair with intensities :math:`y, y'` and predictions per unit reflectivity
:math:`\bar J = g S_b`, :math:`\bar J' = g' S'_{b'}`, both bins gamma with shape :math:`L`
and means :math:`\kappa_q \bar J, \kappa_q \bar J'`, and :math:`p(\kappa_q) \propto 1/\kappa_q`,
the depths enter only through the log-ratio residual

.. math::

    e = \log\frac{y'}{y} - \log\frac{\bar J'}{\bar J}, \qquad
    p_L(e) = \frac{\Gamma(2L)}{\Gamma(L)^2} \frac{e^{L e}}{(1 + e^{e})^{2L}}.

With spikes and the side offset :math:`\mu` (starboard minus port gain constant, sign
:math:`\sigma_m \in \{-1, 0, 1\}`), the factor of passes :math:`a, b` is

.. math::

    \log f_{ab} = \max_\mu \sum_m \log\bigl[(1 - \epsilon_m) p_L(e_m - \sigma_m \mu)
        + \epsilon_m u(e_m - \sigma_m \mu)\bigr],

:math:`\epsilon_m = \epsilon_c + \epsilon_{c'}` and :math:`u` uniform on
:math:`[-E_\mathrm{max}, E_\mathrm{max}]`. :math:`\mu` is profiled out by damped Newton steps,
kept within :math:`\pm E_\mathrm{max}`, and one plain Newton step from the stopped-gradient
value, so gradients with respect to the depths are those of the profile.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from bbf.match.pairs import Pairs
from bbf.ping.factor import DB_TO_LOG_I, bin_ranges
from bbf.ping.theta import Theta

E_MAX = 10.0  # half-width of the uniform outlier density (log units)
NEWTON_ITERS = 30
MAX_STEP = 1.0


def log_ratio_pdf(e, L: float):
    r""":math:`\log p_L(e)`, the density of the log-ratio of two gamma variables of shape L and equal mean."""
    return gammaln(2 * L) - 2 * gammaln(L) + L * e - 2 * L * jnp.logaddexp(0.0, e)


def log_pair(e, eps, L: float):
    r"""Per-pair :math:`\log[(1-\epsilon) p_L(e) + \epsilon\, u(e)]`; u is uniform on :math:`\pm E_\mathrm{max}`."""
    log_u = jnp.where(jnp.abs(e) <= E_MAX, -math.log(2 * E_MAX), -1e3)
    return jnp.logaddexp(jnp.log1p(-eps) + log_ratio_pdf(e, L), jnp.log(eps) + log_u)


def residual(pairs: Pairs, S_a, S_b, y_a, y_b, theta: Theta, dr: float):
    r"""Log-ratio residuals :math:`e_m` (m,) of the pairs; differentiable in S_a, S_b (n, 2, B).

    The prediction per unit reflectivity carries the noise floor in reflectivity units,
    :math:`\bar J = g\,(S_b + N_c e^{-m_c})` with :math:`m_c` the prior mean of
    :math:`\log\kappa`. Selected pairs have :math:`S_b` far above it, so it changes nothing
    there; when the depths move a matched bin into water column or shadow while the pairs
    are held fixed, :math:`\bar J` falls to the noise level instead of zero and the residual
    stays finite.
    """
    r = bin_ranges(jnp.asarray(y_a).shape[-1], dr)
    g = jnp.asarray(theta.gain_db)
    floor = jnp.asarray(np.asarray(theta.noise) * np.exp(-np.asarray(theta.log_kappa_mean)))

    def log_J(S, p, c, b):
        gd = g[c]
        log_g = DB_TO_LOG_I * (gd[:, 0] + gd[:, 1] * r[b] + gd[:, 2] * jnp.log10(r[b]))
        return log_g + jnp.log(S[p, c, b] + floor[c])

    ya = jnp.asarray(y_a)[pairs.ping_a, pairs.side_a, pairs.bin_a]
    yb = jnp.asarray(y_b)[pairs.ping_b, pairs.side_b, pairs.bin_b]
    Ja = log_J(jnp.asarray(S_a), pairs.ping_a, pairs.side_a, pairs.bin_a)
    Jb = log_J(jnp.asarray(S_b), pairs.ping_b, pairs.side_b, pairs.bin_b)
    return jnp.log(yb) - jnp.log(ya) - (Jb - Ja)


def profile_mu(e, sign, eps, L: float):
    r"""Profiled offset :math:`\hat\mu` and the factor's value at it; :math:`\hat\mu` differentiable in e."""
    def F(mu):
        return jnp.sum(log_pair(e - sign * mu, eps, L))

    dF, d2F = jax.grad(F), jax.grad(jax.grad(F))
    has_mixed = jnp.any(sign != 0)

    def step(_, mu):
        h = jnp.maximum(-d2F(mu), 1e-6)
        return jnp.clip(mu + jnp.where(has_mixed, jnp.clip(dF(mu) / h, -MAX_STEP, MAX_STEP), 0.0), -E_MAX, E_MAX)

    mu_star = jax.lax.stop_gradient(jax.lax.fori_loop(0, NEWTON_ITERS, step, 0.0))
    h_star = jnp.maximum(-d2F(mu_star), 1e-6)
    # one plain Newton step for the implicit derivative of the mode; it is tiny at a converged
    # mode, and limited like the damped steps where the profile is flat (most pairs outliers)
    mu_hat = mu_star + jnp.where(has_mixed, jnp.clip(dF(mu_star) / h_star, -MAX_STEP, MAX_STEP), 0.0)
    return mu_hat, F(mu_hat)


def log_match_factor(pairs: Pairs, S_a, S_b, y_a, y_b, theta: Theta, dr: float, L: float | None = None):
    r"""Log match factor :math:`\log f_{ab}` of two passes and the profiled offset :math:`\hat\mu`.

    S_a, S_b: (n, 2, B) unit intensities of every ping of each pass, e.g. from
    `bbf.ping.unit_intensities`; y_a, y_b: their measured intensities. L defaults to the
    ping factor's shape; the paper notes a pair's ratio may call for a larger one.
    """
    if len(pairs) == 0:
        return jnp.asarray(0.0), jnp.asarray(0.0)
    e = residual(pairs, S_a, S_b, y_a, y_b, theta, dr)
    eps_side = np.asarray(theta.eps)
    eps = jnp.asarray(np.clip(eps_side[pairs.side_a] + eps_side[pairs.side_b], 1e-12, 0.5))
    mu_hat, value = profile_mu(e, jnp.asarray(pairs.sign, dtype=e.dtype), eps, theta.L if L is None else L)
    return value, mu_hat
