import numpy as np
import pytest

from bbf.inference import init as I
from bbf.io import Pass
from bbf.ping.theta import Theta

DR, N_BINS = 0.05, 800


def theta():
    return Theta(noise=1.0, L=0.5, eps=0.01, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)


def synthetic_pass(x0, beta, rng, n=40, z=-5.0):
    """Level pass heading north at x = x0 over the plane d = beta . (x, y, 1), z_ref = 0.

    Intensities: noise (Gam(L), mean 1) before the first echo, 1000x brighter after, plus
    transmit ringing (saturated in the first 0.5 m, then decaying over 5 m) and a few
    bright fish in the water column.
    """
    y_pos = np.linspace(-20, 20, n)
    poses = np.column_stack([np.full(n, x0), y_pos, np.full(n, z), np.zeros((n, 3))])
    r = np.arange(N_BINS) * DR
    y = np.zeros((n, 2, N_BINS))
    for t in range(n):
        depth_below = beta[0] * x0 + beta[1] * y_pos[t] + beta[2] + z  # height of the sonar above the plane
        for c in (0, 1):
            mean = np.where(r < depth_below, 1.0, 1000.0)
            mean = mean + 300.0 * np.exp(-r / 5.0)  # transmit ringing decaying over metres, as on port
            mean = np.where(r < 0.5, 1e4, mean)
            vals = rng.gamma(0.5, mean / 0.5)
            fish = rng.random(N_BINS) < 0.002
            y[t, c] = np.where(fish, 1e5, vals)
    return Pass(name=f"x{x0}", poses=poses, y=y, dr=DR, t=np.arange(n) * 0.1)


def test_first_echo_skips_ringing_and_fish():
    rng = np.random.default_rng(0)
    beta = np.array([0.0, 0.0, 20.0])
    ps = synthetic_pass(0.0, beta, rng)
    r0 = I.first_echo(ps, theta())
    assert np.isfinite(r0).all()
    # a fish just before the echo, or dark speckle at its onset, moves a few ping-sides by a
    # block or two; the plane fit is robust to that
    block = theta().block * DR
    err = np.abs(r0 - 15.0)
    assert np.mean(err <= 1.5 * block) >= 0.95
    assert err.max() <= 3 * block
    assert np.median(r0) == pytest.approx(15.0, abs=0.5 * block)


def test_plane_through_first_echoes():
    rng = np.random.default_rng(1)
    beta = np.array([0.05, -0.03, 20.0])
    passes = [synthetic_pass(x0, beta, rng) for x0 in (-10.0, 10.0)]
    mesh = I.mesh_for_passes(passes, h=1.0, buffer=5.0)
    assert mesh.z_ref == pytest.approx(-5.0)
    pts = np.concatenate([I.first_echo_points(p, I.first_echo(p, theta())) for p in passes])
    b = I.plane_fit(pts, mesh)
    # plane in local coordinates: d = b . (x - ox, y - oy, 1), depths measured from z_ref = -5
    ox, oy = mesh.origin
    expect = np.array([beta[0], beta[1], beta[0] * ox + beta[1] * oy + beta[2] - 5.0])
    np.testing.assert_allclose(b[:2], expect[:2], atol=0.01)
    assert b[2] == pytest.approx(expect[2], abs=0.15)
    d0 = I.initial_depths(mesh, b)
    v = mesh.vertices
    np.testing.assert_allclose(d0, b[0] * v[:, 0] + b[1] * v[:, 1] + b[2])


def test_first_echo_ignores_the_other_sides_leak():
    """Starboard sees a bank at 12 m, port the floor at 15 m; port's channel carries starboard's echo 34 dB down."""
    rng = np.random.default_rng(3)
    n = 40
    poses = np.column_stack([np.zeros(n), np.linspace(-20, 20, n), np.full(n, -5.0), np.zeros((n, 3))])
    r = np.arange(N_BINS) * DR
    stbd = np.where(r < 12.0, 1.0, 1e5)
    port = np.where(r < 15.0, 1.0, 1000.0) + 10 ** -3.4 * stbd
    y = np.stack([[rng.gamma(0.5, m / 0.5) for m in (port, stbd)] for _ in range(n)])
    r0 = I.first_echo(Pass(name="leak", poses=poses, y=y, dr=DR, t=np.arange(n) * 0.1), theta())
    block = theta().block * DR
    assert np.median(r0[:, 1]) == pytest.approx(12.0, abs=block)
    assert np.median(r0[:, 0]) == pytest.approx(15.0, abs=block)  # not the leak at 12 m


def test_start_points_take_the_later_side():
    """The up-slope side's echo is the bank's nearest point; the later one bounds the depth below the sonar."""
    ps = Pass(name="p", poses=np.array([[1.0, 2.0, -5.0, 0, 0, 0]]), y=np.ones((1, 2, 4)), dr=DR, t=np.zeros(1))
    pts = I.first_echo_points(ps, np.array([[15.0, 12.0]]))
    np.testing.assert_allclose(pts, [[1.0, 2.0, -20.0]])


def test_bank_points_make_a_start_that_honours_both_first_echoes():
    """Over a floor with a bank on starboard, the points lie on the bank and the start's nearest ranges match both echoes."""
    import jax.numpy as jnp

    from bbf import mesh as M
    from bbf.ping.first_return import TAU, _batched
    from bbf.prior import Prior
    m = M.build(-40.0, 40.0, -15.0, 15.0, h=0.5, origin=(0.0, 0.0), z_ref=0.0)
    depth, slope = 20.0, np.tan(np.radians(35.0))
    n = 30
    # heading north at x = 0: starboard is +x; the bank rises from the point below the sonar
    poses = np.column_stack([np.zeros(n), np.linspace(-7, 7, n), np.zeros(n), np.zeros((n, 3))])
    r_bank = depth * np.cos(np.radians(35.0))
    r0 = np.column_stack([np.full(n, depth), np.full(n, r_bank)])
    ps = Pass(name="a", poses=poses, y=np.ones((n, 2, 4)), dr=0.1, t=np.arange(n) * 0.1)
    bank = I.bank_points(ps, r0)
    assert len(bank) > 5 * n and np.all(bank[:, 0] > 0)
    np.testing.assert_allclose(bank[:, 2], -depth + slope * bank[:, 0], atol=1e-9)
    assert bank[:, 0].max() == pytest.approx(r_bank * np.sin(np.radians(35.0)))
    assert len(I.bank_points(ps, np.column_stack([r0[:, 0], r0[:, 0] - 0.3]))) == 0  # within the gap: no bank
    up = I.bank_points(ps, r0, top=2.0)  # on up the same plane, to 2 m below the sonar
    np.testing.assert_allclose(up[:, 2], -depth + slope * up[:, 0], atol=1e-9)
    assert up[:, 2].max() == pytest.approx(-2.0) and len(up) > len(bank)
    stray = r0.copy()
    stray[10] = [12.0, np.nan]  # one ping's detections off: port far too early, starboard missing
    np.testing.assert_allclose(I.bank_points(ps, stray), bank)
    prior = Prior(m, 5.0, 8.0, mu_beta=np.array([depth]), Sigma_beta=np.array([[100.0]]), alpha=3)
    u = jnp.arange(0.025, 25.0, 0.05)
    res = {}
    for name, pts in (("level", I.first_echo_points(ps, r0)),
                      ("bank", np.concatenate([I.first_echo_points(ps, r0), bank]))):
        d = I.depths_from_points(m, pts, prior)
        rt = np.stack([np.asarray(_batched(m, c, TAU)(jnp.asarray(d), jnp.asarray(poses), u)[0]) for c in (0, 1)], axis=1)
        res[name] = np.median((r0 - rt)[5:-5], axis=0)
    assert abs(res["level"][0]) < 0.3 and res["level"][1] < -3.0  # the level start misses starboard's echo by metres
    assert np.all(np.abs(res["bank"]) < 0.5)
