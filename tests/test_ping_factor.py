import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from bbf import mesh as M
from bbf.ping import factor as F
from bbf.ping import render as Rn
from bbf.ping.theta import PORT, STARBOARD, Theta

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 800, 0.05


def make_theta(eps=0.02, noise=1e-7, L=1.0, std=1.0):
    return Theta(noise=noise, L=L, eps=eps, eta=1e-3, zeta=1e-3, ell=1.0,
                 log_kappa_mean=0.0, log_kappa_std=std, gain_db=[[0.0, 0.1, 2.0], [0.0, 0.05, 3.0]])


def simulate(theta, side, S, u, rng):
    """Bins drawn from the mixture at :math:`u = \\log\\kappa`."""
    I_bar, I_sp = (np.asarray(a) for a in F.predicted(jnp.asarray(S), u, theta, side, DR))
    spike = rng.random(S.shape) < theta.at(side).eps
    main = gamma_dist.rvs(theta.L, scale=I_bar / theta.L, random_state=rng)
    return np.where(spike, gamma_dist.rvs(1.0, scale=I_sp, random_state=rng), main)


def quadrature(theta, side, y, S, use=None):
    """Reference: :math:`\\log \\int e^{F(u)} du` on a fine grid around the mode."""
    use = np.ones(S.shape, bool) if use is None else use
    Fu, _, _ = F._objective(theta, side, DR, jnp.asarray(y), jnp.asarray(S), jnp.asarray(use))
    u_hat, H = F.mode(theta, side, DR, jnp.asarray(y), jnp.asarray(S), jnp.asarray(use))
    sd = 1 / math.sqrt(float(H))
    grid = float(u_hat) + np.linspace(-12, 12, 4001) * sd
    vals = np.asarray(jax.jit(jax.vmap(Fu))(jnp.asarray(grid)))
    return float(np.logaddexp.reduce(vals)) + math.log(grid[1] - grid[0])


def fake_S(rng):
    """A plausible unit-intensity profile: water column, then a decaying seabed echo with texture."""
    r = np.arange(N_BINS) * DR
    S = np.where(r > 15, 1e-4 * (15 / np.maximum(r, 15)) ** 3, 0.0)
    return S * np.exp(0.3 * rng.standard_normal(N_BINS))


def test_log_mixture_matches_scipy():
    rng = np.random.default_rng(0)
    y, I_bar = rng.gamma(1.0, 2.0, 50), rng.uniform(0.5, 3, 50)
    I_sp = I_bar + rng.uniform(1, 10, 50)
    L, eps = 0.7, 0.05
    ref = np.log((1 - eps) * gamma_dist.pdf(y, L, scale=I_bar / L) + eps * gamma_dist.pdf(y, 1.0, scale=I_sp))
    np.testing.assert_allclose(F.log_mixture(jnp.asarray(y), jnp.asarray(I_bar), jnp.asarray(I_sp), L, eps),
                               ref, rtol=1e-10)


@pytest.mark.parametrize("eps,noise,seabed_only", [(1e-12, 1e-12, True), (0.02, 1e-7, False)])
def test_laplace_matches_quadrature(eps, noise, seabed_only):
    """Pure gamma on the seabed bins (the conjugate-like case), then the full mixture with noise."""
    rng = np.random.default_rng(1)
    th = make_theta(eps=eps, noise=noise)
    S = fake_S(rng)
    use = S > 0 if seabed_only else np.ones(N_BINS, bool)
    for side, u_true in ((PORT, 0.4), (STARBOARD, -0.7)):
        y = simulate(th, side, S, u_true, rng)
        lap = float(F.log_ping_side(th, side, DR, jnp.asarray(y), jnp.asarray(S), jnp.asarray(use)))
        assert lap == pytest.approx(quadrature(th, side, y, S, use), abs=0.02)
        u_hat, H = F.mode(th, side, DR, jnp.asarray(y), jnp.asarray(S), jnp.asarray(use))
        assert float(u_hat) == pytest.approx(u_true, abs=4 / math.sqrt(float(H)))


def test_mode_has_implicit_derivative():
    """:math:`\\partial \\hat u / \\partial S` from the Newton step equals re-optimising at a perturbed S."""
    rng = np.random.default_rng(2)
    th = make_theta()
    S = fake_S(rng)
    y = jnp.asarray(simulate(th, PORT, S, 0.3, rng))
    use = jnp.ones(N_BINS, bool)
    dS = jnp.asarray(S * rng.uniform(-0.5, 0.5, N_BINS))

    def u_of(t):
        return F.mode(th, PORT, DR, y, jnp.asarray(S) + t * dS, use)[0]

    _, jv = jax.jvp(u_of, (0.0,), (1.0,))
    eps = 1e-5
    assert float(jv) == pytest.approx(float((u_of(eps) - u_of(-eps)) / (2 * eps)), rel=1e-4)


def test_use_mask_and_nonpositive_bins_are_left_out():
    rng = np.random.default_rng(3)
    th = make_theta()
    S = fake_S(rng)
    y = simulate(th, STARBOARD, S, 0.0, rng)
    use = rng.random(N_BINS) < 0.7
    masked = float(F.log_ping_side(th, STARBOARD, DR, jnp.asarray(y), jnp.asarray(S), jnp.asarray(use)))
    y2 = np.where(use, y, -1.0)  # same bins removed by y <= 0 instead
    assert masked == pytest.approx(float(F.log_ping_side(th, STARBOARD, DR, jnp.asarray(y2), jnp.asarray(S))))


@pytest.fixture(scope="module")
def scene():
    m = M.build(-50.0, 50.0, -50.0, 50.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    bump = 3.0 * np.exp(-((v[:, 0] - 12) ** 2 + (v[:, 1] - 1) ** 2) / (2 * 4.0**2))
    d_true = DEPTH - bump + 0.02 * v[:, 0]
    return m, jnp.asarray(d_true)


POSE = jnp.array([0.0, 0.0, SONAR_Z, 0.01, 0.02, 0.05])
U = Rn.u_grid(N_BINS, DR, du=0.01)


def simulate_ping(m, d, th, rng):
    y = []
    for side in (PORT, STARBOARD):
        S = np.asarray(Rn.unit_intensity(m, d, POSE, side, th, U, N_BINS, DR))
        y.append(simulate(th, side, S, rng.normal(0.0, 0.5), rng))
    return np.stack(y)


def test_factor_gradient_matches_finite_differences(scene):
    m, d = scene
    th = make_theta()
    y = jnp.asarray(simulate_ping(m, d, th, np.random.default_rng(4)))
    v = m.vertices
    dd = jnp.asarray(np.sin(v[:, 0] / 4) * np.cos(v[:, 1] / 6))

    def f(t):
        return F.log_ping_factor(m, d + t * dd, POSE, y, th, U, DR)

    _, jv = jax.jvp(f, (0.0,), (1.0,))
    eps = 1e-6
    assert float(jv) == pytest.approx(float((f(eps) - f(-eps)) / (2 * eps)), rel=1e-3)


def test_factor_peaks_at_true_depths(scene):
    """Along random smooth directions, the factor is largest near the depths that made the data."""
    m, d = scene
    th = make_theta()
    rng = np.random.default_rng(5)
    y = jnp.asarray(simulate_ping(m, d, th, rng))
    v = m.vertices
    f = jax.jit(lambda dd: F.log_ping_factor(m, dd, POSE, y, th, U, DR))
    ts = np.linspace(-1, 1, 21)
    for k in range(3):
        a, b = rng.uniform(3, 8, 2)
        delta = 0.5 * np.sin(v[:, 0] / a + k) * np.cos(v[:, 1] / b)  # up to 0.5 m
        vals = [float(f(d + t * jnp.asarray(delta))) for t in ts]
        assert abs(ts[int(np.argmax(vals))]) <= 0.2
