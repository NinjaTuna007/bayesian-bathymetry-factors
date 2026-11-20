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
    np.testing.assert_allclose(R[:, 0], [1, 0, 0], atol=1e-12)   # yaw :math:`90^\circ` clockwise: east
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
    B = np.exp(-4 * math.log(2) * (beta - th.beta0[PORT]) ** 2 / th.omega[PORT] ** 2)
    expect = B * (D / r_ref) ** 2 * np.exp(-4 * th.alpha * r_ref) / r_ref**3
    np.testing.assert_allclose(r, r_ref, rtol=1e-12)
    np.testing.assert_allclose(a, expect, rtol=1e-10)


def test_echo_zero_for_faces_seen_from_below():
    P = jnp.array([[5.0, 0.0, 0.0]])  # seabed point above the sonar
    a, _ = G.echo(theta(), jnp.array([0, 0, -10.0, 0, 0, 0]), STARBOARD, P, jnp.array([[0.0, 0, 1]]))
    assert float(a[0]) == 0.0


WAVELENGTH = 1480.0 / 450e3


def horizon(m, d, pose, side, u, margin=0.5):
    P, _, _ = G.samples(m, d, pose, side, u)
    return np.asarray(G.horizon_visibility(P, G.pose_frame(pose)[0], u, margin, WAVELENGTH))


def test_knife_edge_values():
    """ITU-R P.526 approximation: 0 dB in the open, about 6 dB at grazing, about 13.9 dB at :math:`\nu = 1`."""
    J = -10 * np.log10(np.asarray(G.knife_edge(jnp.array([-2.0, -0.78, 0.0, 1.0, 10.0]))))
    np.testing.assert_allclose(J[:3], [0.0, 0.0, 6.03], atol=0.02)
    assert J[3] == pytest.approx(13.9, abs=0.1)
    assert J[4] == pytest.approx(6.9 + 20 * np.log10(np.sqrt(9.9**2 + 1) + 9.9), abs=1e-9)


def test_flat_seabed_fully_visible(m):
    u = np.linspace(0.5, 40.0, 400)
    for side in (PORT, STARBOARD):
        V = horizon(m, jnp.full(m.n_vertices, DEPTH), jnp.array([0, 0, SONAR_Z, 0, 0, 0]), side, u)
        assert V.min() > 0.99


def test_shadow_behind_ridge(m):
    v = m.vertices
    ridge = np.where(np.abs(v[:, 0] - 15.0) < 1.0, DEPTH - 10.0, DEPTH)  # 10 m wall at x = 15
    u = np.linspace(0.5, 60.0, 600)
    V = horizon(m, jnp.asarray(ridge), jnp.array([0, 0, SONAR_Z, 0, 0, 0]), STARBOARD, u)
    edge = 15 * D / 5  # the line of sight over the crest (10 m up at x = 15) meets the seabed here
    assert V[u < 13].min() > 0.99
    assert V[(u > 17) & (u < edge - 4)].max() < 0.01   # deep shadow
    assert V[np.argmin(np.abs(u - edge))] == pytest.approx(0.25, abs=0.1)  # grazing: about 6 dB
    assert V[u > edge + 4].min() > 0.99
    # diffraction leaves a small echo that grows towards the shadow edge
    deep = V[(u > 20) & (u < edge - 4)]
    assert deep.min() > 0 and np.all(np.diff(deep) > 0)


def test_gradient_in_deep_shadow(m):
    r"""Deep in shadow, :math:`\partial \log V / \partial d` stays finite: lowering the crest uncovers the seabed."""
    v = m.vertices
    crest = np.abs(v[:, 0] - 15.0) < 1.0
    ridge = jnp.asarray(np.where(crest, DEPTH - 10.0, DEPTH))
    u = np.linspace(0.5, 60.0, 600)
    pose = jnp.array([0, 0, SONAR_Z, 0, 0, 0])
    k = int(np.argmin(np.abs(u - 30.0)))  # well inside the shadow

    def log_V(t):
        P, _, _ = G.samples(m, ridge + t * jnp.asarray(crest, dtype=float), pose, STARBOARD, u)
        return jnp.log(G.horizon_visibility(P, G.pose_frame(pose)[0], u, 0.5, WAVELENGTH)[k])

    g = float(jax.grad(log_V)(0.0))
    assert 0.1 < g < 100  # per metre of crest depth


def test_each_side_follows_its_own_beam(m):
    """With different beams per side, each side's echo is the one rendered with its own beam on both sides."""
    import dataclasses
    th = theta()
    d = jnp.full(m.n_vertices, DEPTH)
    pose = jnp.array([0, 0, SONAR_Z, 0, 0, 0])
    u = jnp.linspace(0.5, 40.0, 80)
    mixed = dataclasses.replace(th, beta0=np.radians([40.0, 60.0]), omega=np.radians([50.0, 70.0]))
    for side, (b0, om) in ((PORT, (40.0, 50.0)), (STARBOARD, (60.0, 70.0))):
        own = dataclasses.replace(th, beta0=np.radians(b0), omega=np.radians(om))
        P, n, _ = G.samples(m, d, pose, side, u)
        a_mixed, _ = G.echo(mixed, pose, side, P, n)
        a_own, _ = G.echo(own, pose, side, P, n)
        np.testing.assert_allclose(a_mixed, a_own, rtol=1e-12)
    P, n, _ = G.samples(m, d, pose, PORT, u)
    assert not np.allclose(G.echo(mixed, pose, PORT, P, n)[0], G.echo(th, pose, PORT, P, n)[0])


def test_off_mesh_samples_cast_no_horizon():
    """A cropped mesh east of the sonar: off-mesh samples have no seabed, so they must not shadow on-mesh ones."""
    def plane(v):
        return DEPTH + 0.3 * v[:, 1]  # tilted along track, so the off-mesh depth guess is wrong
    big = M.build(-5.0, 60.0, -15.0, 15.0, h=1.0, origin=(0.0, 0.0))
    small = M.build(10.0, 60.0, -15.0, 15.0, h=1.0, origin=(0.0, 0.0))
    pose = jnp.array([0.0, 0.0, SONAR_Z, 0.0, 0.0, 0.0])
    u = np.linspace(0.5, 50.0, 500)
    V, inside = {}, {}
    for name, mm in (("big", big), ("small", small)):
        d = jnp.asarray(plane(mm.vertices))
        P, _, inside[name] = G.samples(mm, d, pose, STARBOARD, u)
        V[name] = np.asarray(G.horizon_visibility(P, G.pose_frame(pose)[0], u, 0.5, WAVELENGTH, inside[name]))
    on = np.asarray(inside["small"])
    assert on.any() and (~on).any()
    np.testing.assert_allclose(V["small"][on], V["big"][on], atol=1e-9)
