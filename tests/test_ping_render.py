import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M
from bbf.ping import geometry as G
from bbf.ping import render as Rn
from bbf.ping.theta import PORT, STARBOARD, Theta

DEPTH, SONAR_Z = 20.0, -5.0
D = DEPTH + SONAR_Z
N_BINS, DR = 800, 0.05  # 40 m of slant range


@pytest.fixture(scope="module")
def m():
    return M.build(-60.0, 60.0, -60.0, 60.0, h=1.0, origin=(0.0, 0.0))


@pytest.fixture(scope="module")
def th():
    return Theta(noise=1.0, L=1.0, eps=0.01, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)


LEVEL = jnp.array([0.0, 0.0, SONAR_Z, 0.0, 0.0, 0.0])
U = Rn.u_grid(N_BINS, DR)


def test_split_into_bins_conserves_and_interpolates():
    r = jnp.array([0.0, 0.26, 1.0, 5.0])
    S = Rn.split_into_bins(r, jnp.ones(4), n_bins=11, dr=0.1)
    np.testing.assert_allclose(S.sum(), 3.0)  # the sample at 5 m is past the last bin
    np.testing.assert_allclose(np.asarray(S)[[0, 2, 3, 10]], [1.0, 0.4, 0.6, 1.0], atol=1e-12)


def test_flat_seabed_first_return_and_closed_form(m, th):
    d = jnp.full(m.n_vertices, DEPTH)
    for side in (PORT, STARBOARD):
        S = np.asarray(Rn.unit_intensity(m, d, LEVEL, side, th, U, N_BINS, DR))
        r_b = np.arange(N_BINS) * DR
        # water column before the first return at the sonar's height above the seabed
        first = np.flatnonzero(S > 0)[0]
        assert abs(first * DR - D) <= DR
        # away from the first return, S_b = a(r_b) dq/dr with dq/dr = r / ground on a flat seabed
        sel = (r_b > D + 1) & (r_b < N_BINS * DR - 1)
        ground = np.sqrt(r_b[sel] ** 2 - D**2)
        P = jnp.column_stack([ground, np.zeros_like(ground), np.full_like(ground, SONAR_Z - D)])
        P = P.at[:, 0].multiply(-1.0 if side == PORT else 1.0)
        a, _ = G.echo(th, LEVEL, side, P, jnp.tile(jnp.array([0.0, 0, 1]), (len(ground), 1)))
        np.testing.assert_allclose(S[sel], np.asarray(a) * r_b[sel] / ground, rtol=0.01)


def test_mass_conservation(m, th):
    v = m.vertices
    d = jnp.asarray(DEPTH + 2 * np.sin(v[:, 0] / 5) * np.cos(v[:, 1] / 7))
    pose = jnp.array([1.0, -2.0, SONAR_Z, 0.03, 0.05, 0.4])
    S = Rn.unit_intensity(m, d, pose, STARBOARD, th, U, N_BINS, DR)
    P, n, inside = G.samples(m, d, pose, STARBOARD, U)
    a, r = G.echo(th, pose, STARBOARD, P, n)
    # a sample at x = r / Δr keeps the part of its split that lands on bins 0 .. N-1
    kept = np.clip(N_BINS - np.asarray(r) / DR, 0.0, 1.0)
    w = np.asarray(a * Rn.curve_length(P[:, :2]))
    expect = float(np.sum((w * kept)[np.asarray(inside)]))
    assert float(S.sum() * DR) == pytest.approx(expect, rel=1e-10)


def test_visibility_mask_removes_samples(m, th):
    d = jnp.full(m.n_vertices, DEPTH)
    vis = np.asarray(U) < 20.0
    S_all = Rn.unit_intensity(m, d, LEVEL, PORT, th, U, N_BINS, DR)
    S_vis = Rn.unit_intensity(m, d, LEVEL, PORT, th, U, N_BINS, DR, vis=vis)
    cut = math.hypot(20.0, D) / DR
    np.testing.assert_allclose(S_vis[: int(cut) - 1], S_all[: int(cut) - 1])
    assert float(jnp.abs(S_vis[int(cut) + 2:]).max()) == 0.0


def test_gradient_wrt_depth_and_pose(m, th):
    """Directional derivatives of a smooth summary of S match central differences."""
    v = m.vertices
    d = jnp.asarray(DEPTH + 2 * np.sin(v[:, 0] / 5) * np.cos(v[:, 1] / 7))
    pose = jnp.array([1.0, -2.0, SONAR_Z, 0.03, 0.05, 0.4])
    wts = jnp.asarray(np.random.default_rng(0).standard_normal(N_BINS))

    def f(d, pose):
        return jnp.sum(wts * Rn.unit_intensity(m, d, pose, STARBOARD, th, U, N_BINS, DR)) * 1e4

    rng = np.random.default_rng(1)
    dd = jnp.asarray(np.sin(v[:, 0] / 3 + rng.uniform(0, 6)))  # smooth depth direction
    dp = jnp.asarray(rng.standard_normal(6) * [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    _, jd = jax.jvp(lambda t: f(d + t * dd, pose), (0.0,), (1.0,))
    _, jp = jax.jvp(lambda t: f(d, pose + t * dp), (0.0,), (1.0,))
    eps = 1e-6
    fd_d = (f(d + eps * dd, pose) - f(d - eps * dd, pose)) / (2 * eps)
    fd_p = (f(d, pose + eps * dp) - f(d, pose - eps * dp)) / (2 * eps)
    assert float(jd) == pytest.approx(float(fd_d), rel=1e-3)
    assert float(jp) == pytest.approx(float(fd_p), rel=1e-3)
    # range timing carries gradient: deepening the seabed moves the first return outwards
    g = jax.grad(lambda d: Rn.unit_intensity(m, d, LEVEL, STARBOARD, th, U, N_BINS, DR)[int(D / DR) + 2])(
        jnp.full(m.n_vertices, DEPTH))
    assert float(jnp.abs(g).sum()) > 0
