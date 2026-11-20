import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M
from bbf.ping import geometry as G
from bbf.ping.theta import PORT, STARBOARD, Theta

DEPTH, SONAR_Z = 20.0, -5.0  # seabed 20 m below z_ref = 0, sonar 5 m below z_ref
D = DEPTH + SONAR_Z  # sonar height above a flat seabed


@pytest.fixture(scope="module")
def m():
    return M.build(-60.0, 60.0, -60.0, 60.0, h=1.0, origin=(0.0, 0.0))


def theta():
    return Theta(noise=1.0, L=1.0, eps=0.01, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)


def prototype_rotation(roll, pitch, yaw):
    """Reference: `sonar_pose` from scripts/ping_factor.py, numpy, one pose."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    r_ned = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])
    return np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ r_ned @ np.diag([1.0, -1, -1])


def test_pose_frame_axes():
    s, R = G.pose_frame(jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(s, [1, 2, 3])
    np.testing.assert_allclose(R[:, 0], [0, 1, 0], atol=1e-12)   # forward: north
    np.testing.assert_allclose(R[:, 1], [-1, 0, 0], atol=1e-12)  # port: west
    np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-12)   # up
    _, R = G.pose_frame(jnp.array([0, 0, 0, 0, 0, math.pi / 2]))
    np.testing.assert_allclose(R[:, 0], [1, 0, 0], atol=1e-12)   # yaw 90° clockwise: east
    # outboard and down axes of each side at yaw 0
    np.testing.assert_allclose((R @ G.R_MOUNT[STARBOARD])[:, 1], [0, -1, 0], atol=1e-12)
    _, R = G.pose_frame(jnp.zeros(6))
    np.testing.assert_allclose((R @ G.R_MOUNT[STARBOARD])[:, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose((R @ G.R_MOUNT[PORT])[:, 1], [-1, 0, 0], atol=1e-12)
    for side in (PORT, STARBOARD):
        np.testing.assert_allclose((R @ G.R_MOUNT[side])[:, 2], [0, 0, -1], atol=1e-12)
        assert np.linalg.det(G.R_MOUNT[side]) == pytest.approx(1.0)


def test_pose_frame_matches_prototype():
    rng = np.random.default_rng(0)
    for roll, pitch, yaw in rng.uniform(-math.pi, math.pi, size=(10, 3)) * [0.2, 0.2, 1]:
        _, R = G.pose_frame(jnp.array([0, 0, 0, roll, pitch, yaw]))
        np.testing.assert_allclose(R, prototype_rotation(roll, pitch, yaw), atol=1e-12)


def test_trace_level_pose_flat_seabed(m):
    d = jnp.full(m.n_vertices, DEPTH)
    u = jnp.linspace(0.5, 40.0, 80)
    q = G.trace(m, d, jnp.array([3.0, -2.0, SONAR_Z, 0, 0, 0]), STARBOARD, u)
    np.testing.assert_allclose(q, np.column_stack([3.0 + np.asarray(u), np.full(80, -2.0)]), atol=1e-12)


def test_trace_stays_in_ping_plane(m):
    v = m.vertices
    d = jnp.asarray(DEPTH + 0.2 * v[:, 0] - 0.15 * v[:, 1])  # sloping seabed
    pose = jnp.array([1.0, 1.0, SONAR_Z, 0.05, 0.08, 0.6])
    s, R = G.pose_frame(pose)
    for side in (PORT, STARBOARD):
        P, _, inside = G.samples(m, d, pose, side, jnp.linspace(0.5, 30.0, 60))
        assert bool(inside.all())
        np.testing.assert_allclose((P - s) @ R[:, 0], 0, atol=1e-6)


def test_echo_on_flat_seabed(m):
    th = theta()
    d = jnp.full(m.n_vertices, DEPTH)
    u = jnp.linspace(0.5, 40.0, 80)
    P, n, _ = G.samples(m, d, jnp.array([0, 0, SONAR_Z, 0, 0, 0]), PORT, u)
    a, r = G.echo(th, jnp.array([0, 0, SONAR_Z, 0, 0, 0]), PORT, P, n)
    r_ref = np.hypot(np.asarray(u), D)
    beta = np.arctan2(D, np.asarray(u))
    B = np.exp(-4 * math.log(2) * (beta - th.beta0) ** 2 / th.omega**2)
    expect = B * (D / r_ref) ** 2 * np.exp(-4 * th.alpha * r_ref) / r_ref**3
    np.testing.assert_allclose(r, r_ref, rtol=1e-12)
    np.testing.assert_allclose(a, expect, rtol=1e-10)


def test_echo_zero_for_faces_seen_from_below():
    P = jnp.array([[5.0, 0.0, 0.0]])  # seabed point above the sonar
    a, _ = G.echo(theta(), jnp.array([0, 0, -10.0, 0, 0, 0]), STARBOARD, P, jnp.array([[0.0, 0, 1]]))
    assert float(a[0]) == 0.0


def horizon(m, d, pose, side, u, tau=5e-4, margin=0.5):
    P, _, _ = G.samples(m, d, pose, side, u)
    return np.asarray(G.horizon_visibility(P, G.pose_frame(pose)[0], u, margin, tau))


def test_flat_seabed_fully_visible(m):
    u = np.linspace(0.5, 40.0, 400)
    for side in (PORT, STARBOARD):
        V = horizon(m, jnp.full(m.n_vertices, DEPTH), jnp.array([0, 0, SONAR_Z, 0, 0, 0]), side, u)
        assert V.min() > 0.99


def test_shadow_behind_ridge(m):
    v = m.vertices
    ridge = np.where(np.abs(v[:, 0] - 15.0) < 1.0, DEPTH - 10.0, DEPTH)  # 10 m wall at x = 15
    u = np.linspace(0.5, 60.0, 600)
    pose = jnp.array([0, 0, SONAR_Z, 0, 0, 0])
    for tau in (5e-4, 0.0):
        V = horizon(m, jnp.asarray(ridge), pose, STARBOARD, u, tau=tau)
        assert V[u < 13].min() > 0.99
        # the line of sight over the crest (10 m up at x = 15) meets the seabed at x = 15 · D / 5
        assert V[(u > 16) & (u < 15 * D / 5 - 1)].max() < 0.01
        assert V[u > 15 * D / 5 + 1].min() > 0.99


def test_hard_limit_is_a_step(m):
    v = m.vertices
    ridge = np.where(np.abs(v[:, 0] - 15.0) < 1.0, DEPTH - 10.0, DEPTH)
    V = horizon(m, jnp.asarray(ridge), jnp.array([0, 0, SONAR_Z, 0, 0, 0]), STARBOARD,
                np.linspace(0.5, 60.0, 600), tau=0.0)
    assert set(np.unique(V)) <= {0.0, 1.0}


def test_visibility_gradient_reaches_the_crest(m):
    r"""Lowering the crest uncovers seabed behind it: :math:`\partial V / \partial d_\mathrm{crest} > 0` at the shadow's edge."""
    v = m.vertices
    crest = np.abs(v[:, 0] - 15.0) < 1.0
    ridge = jnp.asarray(np.where(crest, DEPTH - 10.0, DEPTH))
    u = np.linspace(0.5, 60.0, 600)
    pose = jnp.array([0, 0, SONAR_Z, 0, 0, 0])
    edge = int(np.argmin(np.abs(u - 15 * D / 5)))  # sample at the shadow's far edge

    def V_edge(t):
        P, _, _ = G.samples(m, ridge + t * jnp.asarray(crest, dtype=float), pose, STARBOARD, u)
        return G.horizon_visibility(P, G.pose_frame(pose)[0], u, 0.5, 5e-3)[edge]

    assert float(jax.grad(V_edge)(0.0)) > 0
