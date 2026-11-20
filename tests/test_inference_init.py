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
