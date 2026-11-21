r"""Two passes over rippled ground with patchy reflectivity: the shared field against one reflectivity per ping-side.

The data are simulated with an analytic reflectivity :math:`\kappa(q)` (not a field on the mesh), a
finer sample spacing than inference renders with, a gain per ping-side and spikes.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from bbf import match as Mt
from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.inference import objective as O
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import jacobians
from bbf.ping import geometry as G

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR, N_PINGS = 400, 0.1, 9
L = 10.0


def log_reflectivity(q):
    r"""True :math:`\log\kappa(q)`: patches of about :math:`\pm 0.6` on scales from 5 to 40 m."""
    x, y = q[:, 0], q[:, 1]
    return 0.6 * (jnp.sin(x / 3.0 + 1.0) * jnp.cos(y / 5.0) + 0.5 * jnp.cos(x / 7.0 - 0.5) + 0.5 * jnp.sin(1.3 * x + 2.0))


def simulate_side(m, d, pose, side, th, u):
    """Unit intensity of one side with the analytic reflectivity in the integrand."""
    Pk, n, inside = G.samples(m, d, pose, side, u)
    a, r = G.echo(th, pose, side, Pk, n)
    V = G.horizon_visibility(Pk, G.pose_frame(pose)[0], u, th.shadow_margin, th.wavelength, inside)
    w = jnp.where(inside, jnp.exp(log_reflectivity(Pk[:, :2])) * a * V * P.curve_length(Pk[:, :2]), 0.0) / DR
    return P.split_into_bins(r, w, N_BINS, DR)


@pytest.fixture(scope="module")
def fits():
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = DEPTH - 0.15 * np.cos(v[:, 0]) * (1 + 0.3 * np.sin(v[:, 1] / 5.0))  # ripples across the tracks, slopes up to 0.2
    th = P.Theta(noise=1e-7, L=L, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.3)
    u_sim, u = P.u_grid(N_BINS, DR, du=0.02), P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(0)
    poses, ys, labels = [], [], []
    for x0 in (-8.0, 8.0):
        pp = np.column_stack([np.full(N_PINGS, x0), np.linspace(-4, 4, N_PINGS), np.full(N_PINGS, SONAR_Z),
                              np.zeros((N_PINGS, 3))])
        y = np.zeros((N_PINGS, 2, N_BINS))
        for t in range(N_PINGS):
            for c in (0, 1):
                S = simulate_side(m, jnp.asarray(d_true), jnp.asarray(pp[t]), c, th, u_sim)
                Ib, Isp = (np.asarray(a) for a in P.predicted(S, rng.normal(0, 0.2), th, c, DR))
                spike = rng.random(N_BINS) < th.at(c).eps
                y[t, c] = np.where(spike, rng.gamma(1.0, Isp), rng.gamma(L, Ib / L))
        poses.append(pp)
        ys.append(y)
        labels.append(np.stack([[P.block_labels(y[t, c], th, c, DR) for c in (0, 1)] for t in range(N_PINGS)]))
    beta = np.array([0.0, 0.0, DEPTH])
    prior = PR.Prior(m, 4.0, 0.5, mu_beta=beta, Sigma_beta=np.diag([0.01, 0.01, 1.0]))
    plain = O.Problem(mesh=m, prior=prior, theta=th, poses=poses, y=ys, labels=labels, u=u, dr=DR, chunk=N_PINGS)
    field = dataclasses.replace(plain, refl=Mt.Reflectivity(m, 4.0, 0.7))
    d0 = np.full(m.n_vertices, DEPTH)  # the plane through the ripples
    out = {}
    for name, pb in (("plain", plain), ("field", field)):
        d, b, xi = d0.copy(), beta.copy(), np.zeros(m.n_vertices)
        if pb.refl is not None:  # the reflectivity the start implies, as the outer loop does
            _, _, _, xi, _ = gauss_newton(pb, d, b, jacobians(pb, d, xi), xi=xi, hold_depths=True, max_iter=6,
                                          log=lambda _: None)
        for _ in range(3):
            d, b, _, xi, _ = gauss_newton(pb, d, b, jacobians(pb, d, xi), xi=xi, max_iter=6, log=lambda _: None)
        out[name] = (d, xi)
    seen = (np.abs(v[:, 1]) <= 4) & (np.abs(v[:, 0]) <= 28)  # both passes reach it
    return m, d_true, d0, out, seen


def errors(m, d, d_true, seen):
    """RMS error of the depth and of the across-track slope over the seen ground."""
    faces = seen[m.triangles].all(axis=1)
    g = np.asarray(M.face_gradient(m, jnp.asarray(d)) - M.face_gradient(m, jnp.asarray(d_true)))[faces, 0]
    return float(np.sqrt(np.mean((d - d_true)[seen] ** 2))), float(np.sqrt(np.mean(g**2)))


@pytest.mark.slow
def test_ping_factor_alone_turns_reflectivity_into_relief(fits):
    m, d_true, d0, out, seen = fits
    rmse0, _ = errors(m, d0, d_true, seen)
    rmse, slope = errors(m, out["plain"][0], d_true, seen)
    assert rmse > 1.5 * rmse0  # worse than the plane it started from
    assert slope > 0.08


@pytest.mark.slow
def test_shared_field_recovers_the_ripples(fits):
    m, d_true, d0, out, seen = fits
    rmse0, slope0 = errors(m, d0, d_true, seen)
    rmse_p, slope_p = errors(m, out["plain"][0], d_true, seen)
    d, xi = out["field"]
    rmse, slope = errors(m, d, d_true, seen)
    assert rmse < 0.8 * rmse0 and slope < 0.75 * slope0  # relief and slopes recovered from the plane
    assert rmse < 0.5 * rmse_p and slope < 0.8 * slope_p
    xi_true = np.asarray(log_reflectivity(jnp.asarray(m.vertices)))
    assert np.corrcoef(xi[seen], xi_true[seen])[0, 1] > 0.8  # and the reflectivity is what explains the rest
