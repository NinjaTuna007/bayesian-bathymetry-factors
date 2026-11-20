import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M
from bbf import ping as P
from bbf.inference.jacobian import ping_side_jacobian

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 400, 0.1


@pytest.fixture(scope="module")
def scene():
    m = M.build(-45.0, 45.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    crest = np.exp(-((v[:, 0] - 10) ** 2) / (2 * 1.5**2))  # a ridge that casts a shadow on starboard
    d = jnp.asarray(DEPTH - 9.0 * crest - 1.5 * np.sin(v[:, 0] / 6) * np.cos(v[:, 1] / 5))
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)
    return m, d, th, P.u_grid(N_BINS, DR, du=0.02)


@pytest.mark.parametrize("side", [0, 1])
def test_jacobian_matches_jvp_on_level_ping(scene, side):
    """For a level ping the trace does not move with d, so J v equals the exact jvp of the render."""
    m, d, th, u = scene
    pose = jnp.array([0.0, 0.3, SONAR_Z, 0.0, 0.0, 0.0])
    J = ping_side_jacobian(m, d, pose, side, th, u, N_BINS, DR)
    assert J.shape == (N_BINS, m.n_vertices) and J.nnz > 0
    rng = np.random.default_rng(side)
    v = m.vertices
    for k in range(3):
        dv = np.sin(v[:, 0] / rng.uniform(2, 6) + k) * np.cos(v[:, 1] / rng.uniform(2, 6))
        _, jvp = jax.jvp(lambda dd: P.unit_intensity(m, dd, pose, side, th, u, N_BINS, DR), (d,), (jnp.asarray(dv),))
        jvp = np.asarray(jvp)
        np.testing.assert_allclose(J @ dv, jvp, rtol=1e-6, atol=1e-8 * np.abs(jvp).max())


def test_jacobian_close_under_pitch(scene):
    """With pitch and roll the trace's along-track shift is dropped; J stays close to the exact jvp."""
    m, d, th, u = scene
    pose = jnp.array([0.0, 0.3, SONAR_Z, 0.03, 0.05, 0.2])
    J = ping_side_jacobian(m, d, pose, 1, th, u, N_BINS, DR)
    dv = np.sin(m.vertices[:, 0] / 4) * np.cos(m.vertices[:, 1] / 3)
    _, jvp = jax.jvp(lambda dd: P.unit_intensity(m, dd, pose, 1, th, u, N_BINS, DR), (d,), (jnp.asarray(dv),))
    jvp = np.asarray(jvp)
    assert np.linalg.norm(J @ dv - jvp) < 0.1 * np.linalg.norm(jvp)


@pytest.mark.parametrize("side", [0, 1])
def test_jacobian_with_smooth_normals_matches_jvp(scene, side):
    import dataclasses
    m, d, th, u = scene
    th = dataclasses.replace(th, smooth_normals=True)
    pose = jnp.array([0.0, 0.3, SONAR_Z, 0.0, 0.0, 0.0])
    J = ping_side_jacobian(m, d, pose, side, th, u, N_BINS, DR)
    dv = np.sin(m.vertices[:, 0] / 3.7) * np.cos(m.vertices[:, 1] / 2.9)
    _, jvp = jax.jvp(lambda dd: P.unit_intensity(m, dd, pose, side, th, u, N_BINS, DR), (d,), (jnp.asarray(dv),))
    jvp = np.asarray(jvp)
    np.testing.assert_allclose(J @ dv, jvp, rtol=1e-6, atol=1e-8 * np.abs(jvp).max())
