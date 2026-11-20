import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import polygamma

from bbf import match as Mt
from bbf import mesh as M
from bbf import ping as P

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 800, 0.05


def make_theta(eps=0.01):
    return P.Theta(noise=1e-7, L=0.6, eps=eps, eta=1e-3, zeta=1e-3, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)


@pytest.mark.parametrize("L", [0.48, 1.0, 3.0])
def test_log_ratio_pdf_normalised_and_matches_simulation(L):
    x = np.linspace(-60, 60, 200001)
    assert np.trapezoid(np.exp(np.asarray(Mt.log_ratio_pdf(jnp.asarray(x), L))), x) == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(0)
    e = np.log(rng.gamma(L, 1.0, 200000)) - np.log(rng.gamma(L, 1.0, 200000))
    assert e.mean() == pytest.approx(0.0, abs=0.03)
    assert e.var() == pytest.approx(2 * float(polygamma(1, L)), rel=0.02)


def test_profile_mu_recovers_offset_and_its_derivative():
    rng = np.random.default_rng(1)
    L, mu_true, m = 0.6, 0.7, 20000
    sign = rng.integers(-1, 2, m).astype(float)
    e = np.log(rng.gamma(L, 1.0, m)) - np.log(rng.gamma(L, 1.0, m)) + sign * mu_true
    eps = jnp.full(m, 0.02)
    mu_hat, _ = Mt.profile_mu(jnp.asarray(e), jnp.asarray(sign), eps, L)
    assert float(mu_hat) == pytest.approx(mu_true, abs=0.05)
    # same-side pairs only: the offset is not identified and stays at 0
    mu0, _ = Mt.profile_mu(jnp.asarray(e), jnp.zeros(m), eps, L)
    assert float(mu0) == 0.0
    # implicit derivative: the Newton-step mode moves like a re-optimised one
    de = jnp.asarray(rng.standard_normal(m) * 0.1)

    def mu_of(t):
        return Mt.profile_mu(jnp.asarray(e) + t * de, jnp.asarray(sign), eps, L)[0]

    _, jv = jax.jvp(mu_of, (0.0,), (1.0,))
    h = 1e-5
    assert float(jv) == pytest.approx(float((mu_of(h) - mu_of(-h)) / (2 * h)), rel=1e-4)


def test_echo_guard_keeps_the_residual_density():
    """Selecting on :math:`y/\\bar J + y'/\\bar J'` leaves e distributed as :math:`p_L`; selecting on y alone does not."""
    rng = np.random.default_rng(2)
    th = make_theta(eps=1e-9)
    m = 40000
    S = np.exp(rng.uniform(-14, -10, (m, 2, 1)))  # one bin per "ping", both sides
    kappa = np.exp(rng.normal(0, 1.5, m))
    side_a, side_b = rng.integers(0, 2, m), rng.integers(0, 2, m)
    ya = np.zeros((m, 2, 1))
    yb = np.zeros((m, 2, 1))
    ar = np.arange(m)
    ya[ar, side_a, 0] = rng.gamma(th.L, kappa * S[ar, side_a, 0] / th.L)
    yb[ar, side_b, 0] = rng.gamma(th.L, kappa * S[ar, side_b, 0] / th.L)
    proj_a = (ar, side_a, np.zeros(m, int))
    proj_b = (ar, side_b, np.zeros(m, int))
    pairs = Mt.select_pairs(proj_a, proj_b, S, S, ya, yb, th, DR, margin=20.0)
    assert 0.2 * m < len(pairs) < 0.8 * m  # the guard bites
    e = np.asarray(Mt.residual(pairs, S, S, ya, yb, th, DR))
    var = 2 * float(polygamma(1, th.L))
    assert e.mean() == pytest.approx(0.0, abs=0.05)
    assert e.var() == pytest.approx(var, rel=0.05)
    # the naive guard on one bin's intensity biases the ratio
    naive = ya[ar, side_a, 0] > np.median(ya[ar, side_a, 0])
    e_all = np.log(yb[ar, side_b, 0] / S[ar, side_b, 0]) - np.log(ya[ar, side_a, 0] / S[ar, side_a, 0])
    assert e_all[naive].mean() < -0.5


def straight_pass(x0, n=30, yaw=0.0):
    """Level pass heading north (yaw 0) at x = x0, one ping per metre."""
    y = np.arange(n) - n / 2
    return np.column_stack([np.full(n, x0), y, np.full(n, SONAR_Z), np.zeros(n), np.zeros(n), np.full(n, yaw)])


def test_project_on_straight_pass():
    m = M.build(-50.0, 50.0, -50.0, 50.0, h=1.0, origin=(0.0, 0.0))
    d = jnp.full(m.n_vertices, DEPTH)
    poses = straight_pass(0.0)
    q = np.array([[7.0, 0.2], [-7.0, -3.3], [7.0, 40.0], [45.0, 0.0]])
    ping, side, b = Mt.project(m, d, poses, q, N_BINS, DR)
    D = DEPTH + SONAR_Z
    assert ping[0] == 15 and side[0] == 1 and b[0] == round(math.hypot(7, D) / DR)  # starboard is east
    assert ping[1] == 12 and side[1] == 0 and b[1] == round(math.hypot(7, D) / DR)
    assert ping[2] == -1  # beyond the pass's last ping
    assert ping[3] == -1  # beyond the maximum range


@pytest.fixture(scope="module")
def two_passes():
    """Parallel passes 12 m apart over a bump, intensities drawn from the model with :math:`\\kappa = 1`."""
    m = M.build(-50.0, 50.0, -35.0, 35.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    d_true = jnp.asarray(DEPTH - 3.0 * np.exp(-((v[:, 0] - 2) ** 2 + (v[:, 1] - 1) ** 2) / (2 * 4.0**2)))
    th = make_theta()
    u = P.u_grid(N_BINS, DR, du=0.02)
    rng = np.random.default_rng(3)
    passes = []
    for x0 in (-6.0, 6.0):
        poses = straight_pass(x0)
        S = np.asarray(P.unit_intensities(m, d_true, poses, th, u, N_BINS, DR))
        mean = np.exp(np.asarray(P.log_gain(P.bin_ranges(N_BINS, DR), th.at(0).gain_db)))[None, None] * (S + 1e-7)
        y = rng.gamma(th.L, mean / th.L)
        passes.append((poses, S, y))
    return m, d_true, th, u, passes


def test_match_factor_on_two_passes(two_passes):
    m, d_true, th, u, ((pa, Sa, ya), (pb, Sb, yb)) = two_passes
    q = Mt.draw_overlap(pa, pb, N_BINS * DR, 4000, rng=4)
    pairs = Mt.select_pairs(Mt.project(m, d_true, pa, q, N_BINS, DR), Mt.project(m, d_true, pb, q, N_BINS, DR),
                            Sa, Sb, ya, yb, th, DR)
    assert len(pairs) > 300 and (pairs.sign != 0).any() and (pairs.sign == 0).any()
    mask = Mt.matched_mask(pairs, len(pa), N_BINS, "a")
    assert mask.sum() == len(pairs)  # one pair per bin

    @jax.jit
    def f(d):
        S_a = P.unit_intensities(m, d, pa, th, u, N_BINS, DR)
        S_b = P.unit_intensities(m, d, pb, th, u, N_BINS, DR)
        return Mt.log_match_factor(pairs, S_a, S_b, ya, yb, th, DR)

    _, mu = f(d_true)
    # equal gains on both sides: about 400 mixed pairs give a standard error near 0.13
    assert abs(float(mu)) < 0.4
    v = m.vertices
    rng = np.random.default_rng(5)
    for k in range(2):
        a, b = rng.uniform(3, 8, 2)
        delta = jnp.asarray(0.5 * np.sin(v[:, 0] / a + k) * np.cos(v[:, 1] / b))
        ts = np.linspace(-1, 1, 11)
        vals = [float(f(d_true + t * delta)[0]) for t in ts]
        assert abs(ts[int(np.argmax(vals))]) <= 0.2
    # gradient through the rendered unit intensities and the profiled offset. The rendering is
    # piecewise smooth (bin splits, triangle edges, the horizon's crest, the knife-edge and
    # Lambert clips), so a few pairs sit on a kink where finite differences are unstable:
    # check each pair's residual derivative, then the total loosely.
    delta = jnp.asarray(np.sin(v[:, 0] / 4) * np.cos(v[:, 1] / 6))

    @jax.jit
    def e_of(t):
        d = d_true + t * delta
        return Mt.residual(pairs, P.unit_intensities(m, d, pa, th, u, N_BINS, DR),
                           P.unit_intensities(m, d, pb, th, u, N_BINS, DR), ya, yb, th, DR)

    h = 1e-6
    _, je = jax.jvp(e_of, (0.0,), (1.0,))
    fde = (e_of(h) - e_of(-h)) / (2 * h)
    agree = np.abs(np.asarray(je - fde)) <= 1e-3 * (1 + np.abs(np.asarray(fde)))
    assert agree.mean() > 0.97
    _, jv = jax.jvp(lambda t: f(d_true + t * delta)[0], (0.0,), (1.0,))
    fd = (f(d_true + h * delta)[0] - f(d_true - h * delta)[0]) / (2 * h)
    assert float(jv) == pytest.approx(float(fd), rel=0.02)


def test_taken_bins_are_not_paired_again(two_passes):
    """Bins already paired with a third pass are left out, so each bin is in at most one pair."""
    m, d_true, th, _, ((pa, Sa, ya), (pb, Sb, yb)) = two_passes
    q = Mt.draw_overlap(pa, pb, N_BINS * DR, 4000, rng=4)
    proj = (Mt.project(m, d_true, pa, q, N_BINS, DR), Mt.project(m, d_true, pb, q, N_BINS, DR))
    first = Mt.select_pairs(*proj, Sa, Sb, ya, yb, th, DR)
    taken = Mt.matched_mask(first, len(pa), N_BINS, "a")
    second = Mt.select_pairs(*proj, Sa, Sb, ya, yb, th, DR, taken_a=taken)
    assert len(first) > 300
    assert not (Mt.matched_mask(second, len(pa), N_BINS, "a") & taken).any()
