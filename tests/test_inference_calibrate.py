import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from bbf.inference.calibrate import calibrate
from bbf.ping import factor as F
from bbf.ping.theta import Theta

N_BINS, DR = 1000, 0.05


def test_calibrate_recovers_spike_parameters():
    """Data simulated from the bin mixture with known spike level, ringing and decay are fitted back."""
    rng = np.random.default_rng(0)
    r = np.arange(N_BINS) * DR
    S_one = np.where(r > 5, 1e-4 * (5 / np.maximum(r, 5)) ** 3, 0.0)
    m = 60
    true = Theta(noise=1e-7, L=0.6, eps=0.15, eta=[3e-6, 1e-6], zeta=[3e-4, 1e-4], ell=[2.0, 4.0],
                 log_kappa_mean=[0.5, -0.3], log_kappa_std=[0.4, 0.4], gain_db=[[0, 0.1, 2], [0, 0.05, 3]])
    S = np.tile(S_one * np.exp(0.2 * rng.standard_normal((m, 2, N_BINS))), 1)
    u = np.stack([rng.normal(true.at(c).log_kappa_mean, 0.4, m) for c in (0, 1)], axis=1)
    y = np.zeros((m, 2, N_BINS))
    for k in range(m):
        for c in (0, 1):
            I_bar, I_sp = (np.asarray(a) for a in F.predicted(jnp.asarray(S[k, c]), u[k, c], true, c, DR))
            spike = rng.random(N_BINS) < true.at(c).eps
            y[k, c] = np.where(spike, gamma_dist.rvs(1.0, scale=I_sp, random_state=rng),
                               gamma_dist.rvs(true.L, scale=I_bar / true.L, random_state=rng))
    start = Theta(noise=1e-7, L=0.6, eps=0.15, eta=1e-5, zeta=1e-5, ell=1.0, log_kappa_mean=0.0,
                  log_kappa_std=1.0, gain_db=true.gain_db)
    fit, info = calibrate(y, S, np.ones_like(y, bool), start, DR)
    assert info["converged"]
    for name in ("eta", "zeta"):
        np.testing.assert_allclose(np.log(np.asarray(getattr(fit, name))), np.log(np.asarray(getattr(true, name))),
                                   atol=0.3)
    np.testing.assert_allclose(np.asarray(fit.ell), np.asarray(true.ell), rtol=0.3)
    # each u rests on ~900 bins with L = 0.6: standard error ~0.06
    err = info["u"] - u
    assert np.sqrt(np.mean(err**2)) < 0.1 and np.abs(err).max() < 0.35
    np.testing.assert_allclose(np.asarray(fit.log_kappa_mean), u.mean(0), atol=0.1)
    assert fit.L == true.L and np.all(fit.noise == true.noise)


def test_calibrate_respects_bounds_when_unidentified():
    """No ringing in the data: the decay length stays within its bounds and nothing is NaN."""
    rng = np.random.default_rng(1)
    th = Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)
    S = np.full((5, 2, 200), 1e-5)
    y = rng.gamma(0.6, (S + 1e-7) / 0.6)
    fit, _ = calibrate(y, S, np.ones_like(y, bool), th, DR)
    assert np.all(np.isfinite(fit.eta)) and np.all(np.isfinite(fit.zeta))
    ell = np.asarray(fit.ell)
    assert np.all((ell >= 0.2 - 1e-9) & (ell <= 20.0 + 1e-9))
    assert fit.log_kappa_std == pytest.approx(np.array([0.5, 0.5]))


def test_calibrate_fits_shape_and_spike_rate():
    """With `fit_shape`, data of shape 2.5 and spike rates 0.05, 0.1 are fitted back from the values a level would assume."""
    rng = np.random.default_rng(2)
    r = np.arange(N_BINS) * DR
    S_one = np.where(r > 5, 1e-4 * (5 / np.maximum(r, 5)) ** 3, 0.0)
    m = 40
    true = Theta(noise=1e-7, L=2.5, eps=[0.05, 0.1], eta=[3e-5, 1e-5], zeta=1e-12, ell=1.0,
                 log_kappa_mean=0.0, log_kappa_std=0.4)
    S = np.tile(S_one, (m, 2, 1))
    u = rng.normal(0.0, 0.4, (m, 2))
    y = np.zeros((m, 2, N_BINS))
    for c in (0, 1):
        I_bar, I_sp = (np.asarray(a) for a in F.predicted(jnp.asarray(S[:, c]), u[:, c, None], true, c, DR))
        spike = rng.random((m, N_BINS)) < true.at(c).eps
        y[:, c] = np.where(spike, rng.exponential(I_sp), rng.gamma(true.L, I_bar / true.L))
    start = Theta(noise=1e-7, L=13.0, eps=[0.4, 0.4], eta=1e-5, zeta=1e-12, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)
    fit, info = calibrate(y, S, np.ones_like(y, bool), start, DR, fit_shape=True)
    assert info["converged"]
    assert fit.L == pytest.approx(2.5, rel=0.05)
    np.testing.assert_allclose(np.asarray(fit.eps), [0.05, 0.1], rtol=0.2)
    np.testing.assert_allclose(np.log(np.asarray(fit.eta)), np.log([3e-5, 1e-5]), atol=0.3)
    assert np.sqrt(np.mean((info["u"] - u) ** 2)) < 0.05
    # without it the shape and the rates stay, and the fit is far worse
    kept, plain = calibrate(y, S, np.ones_like(y, bool), start, DR)
    assert kept.L == 13.0 and np.all(np.asarray(kept.eps) == 0.4) and plain["nll"] > info["nll"] + 1000.0
