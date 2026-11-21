r"""Refit of the spike parameters and the log-reflectivity prior at the current depths.

At fixed unit intensities S, the spike level :math:`\eta_c`, ringing level :math:`\zeta_c` and
decay length :math:`\ell_c` of each side are fitted by maximum likelihood over the bins the
ping factor scores, jointly with one :math:`u = \log\kappa` per ping and side. The prior on
:math:`\log\kappa` is then set from those :math:`\hat u`. :math:`L`, :math:`N_c`,
:math:`\epsilon_c` and the gain are kept. The outer loop calls this every round, so the fit
improves as the seabed does.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from bbf.ping.factor import background, bin_ranges, log_gain, log_gamma
from bbf.ping.theta import Theta

ELL_BOUNDS = (0.2, 20.0)  # ringing decay length (m), as in the prototype
LEVEL_FLOOR = 1e-3  # spike and ringing levels at least this fraction of the noise


def calibrate(y, S, use, theta: Theta, dr: float, min_log_kappa_std: float = 0.5,
              offset=None) -> tuple[Theta, dict]:
    r"""Fitted copy of `theta` and fit details, from intensities y, unit intensities S and use masks (m, 2, B).

    Returns `theta` with :math:`\eta_c, \zeta_c, \ell_c` replaced and the log-:math:`\kappa` prior set
    to the mean and standard deviation (at least `min_log_kappa_std`) of the fitted
    :math:`\hat u` per side; the dict holds :math:`\hat u` (m, 2), the negative log likelihood and
    whether the optimiser converged. offset (m,): the known shift of each ping's log-:math:`\kappa`
    prior mean (`Problem.kappa_offset`), removed from :math:`\hat u` before its mean and spread.
    """
    y, S, use = (np.asarray(a) for a in (y, S, use))
    m, _, n_bins = y.shape
    ok = use & (y > 0)
    r = np.asarray(bin_ranges(n_bins, dr))
    logg = np.stack([np.asarray(log_gain(r, theta.at(c).gain_db)) for c in (0, 1)])
    g = np.exp(logg)
    noise = np.asarray(theta.noise, dtype=float)
    off = np.zeros(m) if offset is None else np.asarray(offset, dtype=float)
    # background of every bin (m, 2, B): constant per side unless it follows the range or the power
    bg = jnp.asarray(np.stack([np.broadcast_to(np.asarray(background(theta, c, r, off[:, None])), (m, n_bins))
                               for c in (0, 1)], axis=1))
    eps = np.asarray(theta.eps, dtype=float)
    yj, Sj, okj, gj, rj = (jnp.asarray(a) for a in (np.where(ok, y, 1.0), S, ok, g, r))
    L = theta.L

    def nll(x):
        log_eta, log_zeta, log_ell, u = x[0:2], x[2:4], x[4:6], x[6:].reshape(m, 2)
        I_bar = gj * (jnp.exp(u)[..., None] * Sj + bg)
        extra = jnp.exp(log_eta)[:, None] + jnp.exp(log_zeta)[:, None] * jnp.exp(-rj / jnp.exp(log_ell)[:, None])
        I_sp = I_bar + gj * extra
        lp = jnp.logaddexp(jnp.log1p(-jnp.asarray(eps))[:, None] + log_gamma(yj, L, I_bar),
                           jnp.log(jnp.asarray(eps))[:, None] + log_gamma(yj, 1.0, I_sp))
        return -jnp.sum(jnp.where(okj, lp, 0.0))

    f = jax.jit(jax.value_and_grad(nll))
    # start: u from the median ratio of the scored bins; spike levels from the current theta
    u0 = np.zeros((m, 2))
    for c in (0, 1):
        for k in range(m):
            sel = ok[k, c] & (S[k, c] > 0)
            u0[k, c] = (np.log(np.median(y[k, c][sel] / g[c][sel]) / np.median(S[k, c][sel])) if sel.any()
                        else theta.at(c).log_kappa_mean)
    floor = np.log(LEVEL_FLOOR * noise)
    x0 = np.concatenate([np.maximum(np.log(theta.eta), floor), np.maximum(np.log(theta.zeta), floor),
                         np.clip(np.log(theta.ell), *np.log(ELL_BOUNDS)), u0.ravel()])
    bounds = ([(floor[c], None) for c in (0, 1)] * 2 + [tuple(np.log(ELL_BOUNDS))] * 2
              + [(None, None)] * (2 * m))
    res = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in f(jnp.asarray(x))), x0, jac=True,
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
    x = res.x
    u_hat = x[6:].reshape(m, 2)
    u_rel = u_hat - (0.0 if offset is None else np.asarray(offset, dtype=float)[:, None])
    scored = ok.any(axis=2)  # ping-sides with at least one scored bin
    mean = np.array([u_rel[scored[:, c], c].mean() if scored[:, c].any() else theta.at(c).log_kappa_mean
                     for c in (0, 1)])
    std = np.array([max(u_rel[scored[:, c], c].std(), min_log_kappa_std) if scored[:, c].sum() > 1
                    else theta.at(c).log_kappa_std for c in (0, 1)])
    fitted = dataclasses.replace(theta, eta=np.exp(x[0:2]), zeta=np.exp(x[2:4]), ell=np.exp(x[4:6]),
                                 log_kappa_mean=mean, log_kappa_std=std)
    return fitted, {"u": u_hat, "nll": float(res.fun), "converged": bool(res.success),
                    "nll_per_bin": float(res.fun) / max(int(ok.sum()), 1)}
