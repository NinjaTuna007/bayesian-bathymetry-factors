import numpy as np
import pytest

from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import Pass
from bbf.ping.theta import Theta


def a_pass(y, dr=0.1):
    n = y.shape[0]
    return Pass(name="p", poses=np.zeros((n, 6)), y=y, dr=dr, t=np.arange(n), bathy_ping=np.zeros(0, np.int64),
                bathy_xyz=np.zeros((0, 3)))


def test_coarsen_pass_centres_and_means():
    B = 40
    y = np.broadcast_to(np.arange(B, dtype=float), (2, 2, B)).copy()  # intensity equals the fine bin index
    c = coarsen_pass(a_pass(y), 5)
    assert c.dr == pytest.approx(0.5)
    # coarse bin j averages fine bins j*5-2 .. j*5+2, whose mean is j*5 (bin 0 is clipped to 0..2)
    np.testing.assert_allclose(c.y[0, 0, 1:], 5.0 * np.arange(1, c.n_bins))
    assert c.y[0, 0, 0] == pytest.approx(1.0)
    assert c.n_bins == (B - 1 - 2) // 5 + 1
    with pytest.raises(ValueError):
        coarsen_pass(a_pass(y), 4)


def test_coarse_statistics_are_gamma_kL():
    rng = np.random.default_rng(0)
    L, k = 0.6, 9
    y = rng.gamma(L, 1.0 / L, (1, 2, 9 * 20000))
    c = coarsen_pass(a_pass(y), k).y[0, 0, 1:]  # whole blocks only
    assert c.mean() == pytest.approx(1.0, abs=0.01)
    assert 1.0 / c.var() == pytest.approx(k * L, rel=0.05)  # variance of Gam(kL) with mean 1
    th = coarsen_theta(Theta(noise=1.0, L=L, eps=0.02, eta=9.0, zeta=18.0, ell=1.0, log_kappa_mean=0.0,
                             log_kappa_std=1.0), k)
    assert th.L == pytest.approx(k * L) and th.block == 1
    np.testing.assert_allclose(np.asarray(th.eps), 1 - 0.98**k)
    np.testing.assert_allclose(np.asarray(th.eta), 1.0)
