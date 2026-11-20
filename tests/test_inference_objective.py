import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import match as Mt
from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.inference import objective as O

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 400, 0.1


def straight_pass(x0, n=6):
    y = np.linspace(-3, 3, n)
    return np.column_stack([np.full(n, x0), y, np.full(n, SONAR_Z), np.zeros(n), np.zeros(n), np.zeros(n)])


@pytest.fixture(scope="module")
def problem():
    m = M.build(-45.0, 45.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    d_true = jnp.asarray(DEPTH - 3.0 * np.exp(-((v[:, 0] - 2) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(0)
    poses, ys, labels = [], [], []
    for x0 in (-6.0, 6.0):
        pp = straight_pass(x0)
        S = np.asarray(P.unit_intensities(m, d_true, pp, th, u, N_BINS, DR))
        y = rng.gamma(th.L, (S + 1e-7) / th.L)
        poses.append(pp)
        ys.append(y)
        labels.append(np.stack([[P.block_labels(y[t, c], th, c, DR) for c in (0, 1)] for t in range(len(pp))]))
    q = Mt.draw_overlap(poses[0], poses[1], N_BINS * DR, 600, rng=1)
    S0, S1 = (np.asarray(P.unit_intensities(m, d_true, pp, th, u, N_BINS, DR)) for pp in poses)
    pairs = Mt.select_pairs(Mt.project(m, d_true, poses[0], q, N_BINS, DR),
                            Mt.project(m, d_true, poses[1], q, N_BINS, DR), S0, S1, ys[0], ys[1], th, DR)
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=np.array([0.0, 0.0, DEPTH]), Sigma_beta=np.diag([0.01, 0.01, 25.0]))
    pb = O.Problem(mesh=m, prior=prior, theta=th, poses=poses, y=ys, labels=labels, u=u, dr=DR,
                   pairs={(0, 1): pairs}, chunk=4)
    d0 = d_true + 0.3 * jnp.asarray(np.sin(v[:, 0] / 5))
    return pb, d0, np.array([0.0, 0.0, DEPTH])


def direct(pb, d):
    """The same data term written out factor by factor, with no chunking."""
    total = 0.0
    uses = pb.uses()
    S = [P.unit_intensities(pb.mesh, d, pp, pb.theta, pb.u, pb.n_bins, pb.dr) for pp in pb.poses]
    for p in range(2):
        for t in range(len(pb.poses[p])):
            for c in (0, 1):
                y, s, use = jnp.asarray(pb.y[p][t, c]), S[p][t, c], jnp.asarray(uses[p][t, c])
                total = total + P.log_ping_side(pb.theta, c, pb.dr, y, s, use)
                u_hat, _ = P.mode(pb.theta, c, pb.dr, y, s, use)
                total = total + P.log_shadow_side(pb.theta, c, pb.labels[p][t, c], s, u_hat)
    v, _ = Mt.log_match_factor(pb.pairs[(0, 1)], S[0], S[1], pb.y[0], pb.y[1], pb.theta, pb.dr)
    return total + v


def test_value_and_grad_match_direct_composition(problem):
    pb, d0, beta = problem
    assert len(pb.pairs[(0, 1)]) > 20
    value, (gd, gb), details = O.value_and_grad(pb, d0, beta)
    prior = pb.prior.log_density(np.asarray(d0), beta)
    assert value == pytest.approx(prior + float(direct(pb, d0)), rel=1e-10)
    # data gradient against plain reverse mode through the whole composition
    g_direct = np.asarray(jax.grad(lambda d: direct(pb, d))(d0))
    gd_prior, gb_prior = pb.prior.grad(np.asarray(d0), beta)
    np.testing.assert_allclose(gd - gd_prior, g_direct, rtol=1e-6, atol=1e-6 * np.abs(g_direct).max())
    np.testing.assert_allclose(gb, gb_prior)
    assert details["u"][0].shape == (6, 2) and len(details["mu"]) == 1
