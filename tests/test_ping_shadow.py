import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from bbf import mesh as M
from bbf.ping import factor as F
from bbf.ping import render as Rn
from bbf.ping import shadow as Sh
from bbf.ping.theta import STARBOARD, Theta

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 800, 0.05
U = Rn.u_grid(N_BINS, DR, du=0.01)
LEVEL = jnp.array([0.0, 0.0, SONAR_Z, 0.0, 0.0, 0.0])


def make_theta(eps=0.02):
    return Theta(noise=1e-7, L=0.5, eps=eps, eta=1e-3, zeta=1e-3, ell=1.0,
                 log_kappa_mean=0.0, log_kappa_std=1.0, gain_db=[[0.0, 0.1, 2.0], [0.0, 0.05, 3.0]])


def simulate(theta, side, S, log_kappa, rng):
    I_bar, I_sp = (np.asarray(a) for a in F.predicted(jnp.asarray(S), log_kappa, theta, side, DR))
    spike = rng.random(S.shape) < theta.at(side).eps
    main = gamma_dist.rvs(theta.L, scale=I_bar / theta.L, random_state=rng)
    return np.where(spike, gamma_dist.rvs(1.0, scale=I_sp, random_state=rng), main)


def test_false_alarm_rate_on_noise():
    th = make_theta(eps=1e-12)
    rng = np.random.default_rng(0)
    n = 8 * 20000
    y = simulate(th, STARBOARD, np.zeros(n), 0.0, rng)
    rate = Sh.block_labels(y, th, STARBOARD, DR).mean()
    assert rate == pytest.approx(th.label_fa, abs=4 * np.sqrt(th.label_fa / 20000))


def test_detection_matches_simulation_and_rises_with_snr():
    """With S constant within blocks the moment-matched block mean is exact."""
    th = make_theta()
    rng = np.random.default_rng(1)
    n_blocks = 4000
    for snr in (0.0, 0.5, 2.0, 8.0):
        S = np.full(8 * n_blocks, snr * th.at(STARBOARD).noise)  # :math:`\kappa = 1`
        y = simulate(th, STARBOARD, S, 0.0, rng)
        observed = Sh.block_labels(y, th, STARBOARD, DR).mean()
        predicted = float(Sh.lit_probability(jnp.asarray(S), 0.0, th, STARBOARD).mean())
        assert observed == pytest.approx(predicted, abs=0.03)
    p = [float(Sh.lit_probability(jnp.full(8, s * 1e-7), 0.0, th, STARBOARD)[0]) for s in (0, 1, 3, 10, 30)]
    assert np.all(np.diff(p) > 0)


@pytest.fixture(scope="module")
def wall_scene():
    """True seabed: a wall up to a plateau 3 m below the sonar. Model: the same wall, but
    the seabed drops back behind the crest, so the model puts the plateau in shadow."""
    m = M.build(-40.0, 60.0, -30.0, 30.0, h=1.0, origin=(0.0, 0.0))
    x = m.vertices[:, 0]
    wall = np.clip((x - 5.0) / 5.0, 0.0, 1.0)  # 0 before x = 5, 1 after x = 10
    d_true = DEPTH - 12.0 * wall
    behind = (x > 12.0)
    d_model = np.where(behind, DEPTH, d_true)
    return m, jnp.asarray(d_true), jnp.asarray(d_model), behind


def test_shadow_factor_raises_seabed_behind_crest(wall_scene):
    m, d_true, d_model, behind = wall_scene
    th = make_theta()
    rng = np.random.default_rng(2)
    labels, lk = [], jnp.zeros(2)
    for side in (0, 1):
        S = np.asarray(Rn.unit_intensity(m, d_true, LEVEL, side, th, U, N_BINS, DR))
        labels.append(Sh.block_labels(simulate(th, side, S, 0.0, rng), th, side, DR))
    labels = np.stack(labels)

    def f(d):
        return Sh.log_shadow_factor(m, d, LEVEL, labels, th, U, DR, N_BINS, lk)

    assert float(f(d_true)) > float(f(d_model))
    g = np.asarray(jax.grad(f)(d_model))
    # shallower seabed behind the crest explains the lit blocks: the gradient there is negative
    assert g[behind].sum() < 0
