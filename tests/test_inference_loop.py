import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.loop import infer
from bbf.io import Pass

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 400, 0.1


def simulate_pass(m, d, poses, th, u, rng, name):
    S = np.asarray(P.unit_intensities(m, d, poses, th, u, N_BINS, DR))
    y = np.zeros_like(S)
    for t in range(len(poses)):
        for c in (0, 1):
            I_bar, I_sp = (np.asarray(a) for a in P.predicted(jnp.asarray(S[t, c]), rng.normal(0, 0.3), th, c, DR))
            spike = rng.random(N_BINS) < th.at(c).eps
            y[t, c] = np.where(spike, gamma_dist.rvs(1.0, scale=I_sp, random_state=rng),
                               gamma_dist.rvs(th.L, scale=I_bar / th.L, random_state=rng))
    return Pass(name=name, poses=poses, y=y, dr=DR, t=np.arange(len(poses)) * 0.1)


@pytest.fixture(scope="module")
def run():
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(DEPTH - 3.0 * np.exp(-((v[:, 0] - 3) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)) + 0.03 * v[:, 0])
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(0)
    passes = []
    for x0 in (-6.0, 6.0):
        yy = np.linspace(-4, 4, 9)
        poses = np.column_stack([np.full(9, x0), yy, np.full(9, SONAR_Z), np.zeros((9, 3))])
        passes.append(simulate_pass(m, jnp.asarray(d_true), poses, th, u, rng, f"x{x0}"))
    pts = np.concatenate([I.first_echo_points(p, I.first_echo(p, th)) for p in passes])
    beta0 = I.plane_fit(pts, m)
    d0 = I.initial_depths(m, beta0)
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=beta0, Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    out = infer(passes, m, prior, th, d0, beta0, u, rounds=2, inner=8, calib_every=3, chunk=4,
                log=lambda s: None)
    seen = (np.abs(v[:, 1]) <= 4) & (np.abs(v[:, 0]) <= 30)
    return d_true, d0, out, seen


@pytest.mark.slow
def test_inference_ascends(run):
    _, _, out, _ = run
    for r in out["history"]:
        e = [h["ell"] for h in r["gn"]]
        assert len(e) > 2 and np.all(np.diff(e) > 0)  # every accepted step ascends


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="Gauss-Newton stops in a local optimum: the depth likelihood is rough at "
                   "bin scale, so steps fit speckle-scale detail and miss the large-scale tilt of the start plane")
def test_inference_recovers_bump_from_plane(run):
    d_true, d0, out, seen = run
    rmse0 = float(np.sqrt(np.mean((d0[seen] - d_true[seen]) ** 2)))
    rmse = float(np.sqrt(np.mean((out["d"][seen] - d_true[seen]) ** 2)))
    assert rmse < 0.5 * rmse0


@pytest.mark.slow
def test_trend_search_crosses_to_the_true_tilt():
    """From a tilted start on a featureless slope, the global trend search finds the true slope and the
    Laplace standard deviations then cover the errors; Gauss-Newton alone stays at the start's tilt."""
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = 20.0 + 0.03 * v[:, 0]
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(0)
    passes = []
    for x0 in (-6.0, 6.0):
        poses = np.column_stack([np.full(9, x0), np.linspace(-4, 4, 9), np.full(9, SONAR_Z), np.zeros((9, 3))])
        passes.append(simulate_pass(m, jnp.asarray(d_true), poses, th, u, rng, f"x{x0}"))
    seen = (np.abs(v[:, 1]) <= 4) & (np.abs(v[:, 0]) <= 30)
    b0 = np.array([0.08, 0.0, 20.0])
    d0 = b0[0] * v[:, 0] + b0[2]
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=b0, Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    out = infer(passes, m, prior, th, d0, b0, u, rounds=1, inner=8, calib_every=3, chunk=4,
                laplace_samples=100, trend_search={"slopes_x": np.arange(-0.02, 0.131, 0.025), "steps": 4},
                log=lambda s: None)
    err = out["d"] - d_true
    assert out["beta"][0] == pytest.approx(0.03, abs=0.01)
    assert float(np.sqrt(np.mean(err[seen] ** 2))) < 0.3
    assert np.median(np.abs(err[seen]) / out["sd_d"][seen]) < 2.0


@pytest.mark.slow
def test_inference_with_free_sonar_parameters():
    """Two rounds with psi free from a wrong beam: every step ascends and psi moves into theta."""
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(DEPTH - 3.0 * np.exp(-((v[:, 0] - 3) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(1)
    passes = []
    for x0 in (-6.0, 6.0):
        poses = np.column_stack([np.full(9, x0), np.linspace(-4, 4, 9), np.full(9, SONAR_Z), np.zeros((9, 3))])
        passes.append(simulate_pass(m, jnp.asarray(d_true), poses, th, u, rng, f"x{x0}"))
    import dataclasses
    wrong = dataclasses.replace(th, beta0=np.radians([56.0, 44.0]))
    psi0 = P.psi_of(wrong)
    sd = np.array([[np.radians(15), np.radians(20), 0.2]] * 2)
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=np.array([0.0, 0.0, DEPTH]), Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    out = infer(passes, m, prior, wrong, jnp.asarray(d_true), np.array([0.0, 0.0, DEPTH]), u, rounds=2, inner=4,
                calib_every=3, chunk=4, psi_prior=(psi0, sd), log=lambda s: None)
    for r in out["history"]:
        e = [h["ell"] for h in r["gn"]]
        assert np.all(np.diff(e) > 0)
    assert np.all(np.isfinite(out["psi"])) and not np.allclose(out["psi"], psi0)
    np.testing.assert_allclose(P.psi_of(out["theta"]), out["psi"])
    np.testing.assert_allclose(P.psi_of(out["problem"].theta), out["psi"])


@pytest.mark.slow
def test_inference_with_reflectivity_field():
    """The outer loop with the shared field: fitted at the start depths, then stepped with them; every step ascends."""
    from bbf import match as Mt
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(DEPTH - 0.15 * np.cos(v[:, 0]))
    xi_true = jnp.asarray(0.6 * np.sin(v[:, 0] / 3.0 + 1.0) * np.cos(v[:, 1] / 5.0))
    th = P.Theta(noise=1e-7, L=5.0, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(2)
    passes = []
    for x0 in (-8.0, 8.0):
        poses = np.column_stack([np.full(9, x0), np.linspace(-4, 4, 9), np.full(9, SONAR_Z), np.zeros((9, 3))])
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d_true), poses, th, u, N_BINS, DR, xi_true))
        y = rng.gamma(th.L, (S + 1e-7) / th.L)
        passes.append(Pass(name=f"x{x0}", poses=poses, y=y, dr=DR, t=np.arange(9) * 0.1))
    beta = np.array([0.0, 0.0, DEPTH])
    prior = PR.Prior(m, 4.0, 0.5, mu_beta=beta, Sigma_beta=np.diag([0.01, 0.01, 1.0]))
    out = infer(passes, m, prior, th, np.full(m.n_vertices, DEPTH), beta, u, rounds=2, inner=4, calib_every=3, chunk=9,
                refl=Mt.Reflectivity(m, 4.0, 0.7), laplace_samples=30, log=lambda s: None)
    for r in out["history"]:
        assert np.all(np.diff([h["ell"] for h in r["gn"]]) > 0)
    seen = (np.abs(v[:, 1]) <= 4) & (np.abs(v[:, 0]) <= 28)
    assert np.corrcoef(out["xi"][seen], np.asarray(xi_true)[seen])[0, 1] > 0.7
    assert out["sd_xi"].shape == out["xi"].shape and np.all(np.isfinite(out["sd_d"]))
    assert out["problem"].refl is not None
