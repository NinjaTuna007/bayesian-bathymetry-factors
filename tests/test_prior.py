import math

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.special import kv

from bbf import mesh as M
from bbf import prior as P


def fem_assembly(m):
    """Reference P1 assembly for any triangulation: consistent-mass row sums and stiffness."""
    n = m.n_vertices
    C, S = np.zeros(n), np.zeros((n, n))
    for t in m.triangles:
        p = m.vertices[t]
        B = np.array([p[1] - p[0], p[2] - p[0]]).T
        area = abs(np.linalg.det(B)) / 2
        G = np.linalg.inv(B).T @ np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])  # (2, 3) hat gradients
        S[np.ix_(t, t)] += area * G.T @ G
        C[t] += area / 3
    return C, S


@pytest.fixture
def small():
    return M.build(0.0, 6.0, 0.0, 5.0, h=1.0)


def test_mass_and_stiffness_match_fem(small):
    C, S = fem_assembly(small)
    np.testing.assert_allclose(P.mass(small).diagonal(), C, atol=1e-12)
    np.testing.assert_allclose(P.stiffness(small).toarray(), S, atol=1e-12)


def test_stiffness_annihilates_constants_and_precision_shape(small):
    np.testing.assert_allclose(P.stiffness(small) @ np.ones(small.n_vertices), 0, atol=1e-12)
    L = P.precision(small, rho=3.0, sigma=0.5).toarray()
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    assert np.linalg.eigvalsh(L).min() > 0


def test_precision_bandwidth():
    m = M.build(0.0, 20.0, 0.0, 20.0, h=1.0)
    L = P.precision(m, 5.0, 1.0).tocsr()
    nnz = np.diff(L.indptr)
    assert nnz.max() == 19  # vertex, 6 neighbours, 12 second neighbours


def make_prior(m, rho, sigma):
    return P.Prior(m, rho, sigma, mu_beta=np.zeros(3), Sigma_beta=np.eye(3))


def covariance_column(prior, i):
    e = np.zeros(prior.mesh.n_vertices)
    e[i] = 1.0
    return np.asarray(spla.spsolve(prior.Lambda, e))


def nearest(m, xy):
    return int(np.argmin(np.linalg.norm(m.vertices - xy, axis=1)))


def test_matern_variance_and_range():
    rho, sigma = 10.0, 0.5
    m = M.build(-50.0, 50.0, -50.0, 50.0, h=rho / 10, origin=(0.0, 0.0))
    pr = make_prior(m, rho, sigma)
    i = nearest(m, (0.0, 0.0))
    cov = covariance_column(pr, i)
    assert cov[i] == pytest.approx(sigma**2, rel=0.05)
    k = math.sqrt(8) / rho
    expect = k * rho * kv(1, k * rho)  # Matérn :math:`\nu = 1` correlation at distance :math:`\rho`, :math:`\approx 0.14`
    for xy in [(rho, 0.0), (0.0, rho), (-rho * 0.6, -rho * 0.8)]:
        j = nearest(m, xy)
        r = np.linalg.norm(m.vertices[j] - m.vertices[i])
        assert cov[j] / cov[i] == pytest.approx(k * r * kv(1, k * r), abs=0.02)
    assert expect == pytest.approx(0.14, abs=0.005)


def test_boundary_inflation_and_buffer():
    rho, sigma = 10.0, 0.5
    m = M.build(-30.0, 30.0, -30.0, 30.0, h=rho / 10, origin=(0.0, 0.0))
    pr = make_prior(m, rho, sigma)
    edge = nearest(m, (0.0, m.vertices[:, 1].min()))
    inner = nearest(m, (0.0, m.vertices[:, 1].min() + 2 * rho))
    assert covariance_column(pr, edge)[edge] > 1.5 * sigma**2
    assert covariance_column(pr, inner)[inner] == pytest.approx(sigma**2, rel=0.1)


def test_sample_covariance():
    rho, sigma = 4.0, 0.5
    m = M.build(-10.0, 10.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    pr = make_prior(m, rho, sigma)
    rng = np.random.default_rng(0)
    U = np.stack([P.sample(pr, rng, beta=np.zeros(3))[0] for _ in range(4000)])
    emp = np.cov(U.T)
    ref = np.linalg.inv(pr.Lambda.toarray())
    i = nearest(m, (0.0, 0.0))
    np.testing.assert_allclose(emp[i], ref[i], atol=0.02)


def test_grad_and_joint_precision(small):
    pr = P.Prior(small, 3.0, 0.5, mu_beta=np.array([0.1, -0.2, 15.0]), Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    rng = np.random.default_rng(4)
    d, beta = rng.standard_normal(small.n_vertices) + 15, np.array([0.05, 0.02, 14.0])
    gd, gb = pr.grad(d, beta)
    eps = 1e-6
    for i in range(0, small.n_vertices, 7):
        e = np.zeros_like(d)
        e[i] = eps
        fd = (pr.log_density(d + e, beta) - pr.log_density(d - e, beta)) / (2 * eps)
        assert gd[i] == pytest.approx(fd, rel=1e-6, abs=1e-6)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        fd = (pr.log_density(d, beta + e) - pr.log_density(d, beta - e)) / (2 * eps)
        assert gb[i] == pytest.approx(fd, rel=1e-6, abs=1e-6)
    # quadratic: :math:`\log p(z + \delta) - \log p(z) = g^\top \delta - \tfrac{1}{2} \delta^\top H \delta` exactly
    Q = pr.joint_precision()
    z = np.concatenate([d, beta])
    dz = rng.standard_normal(z.size) * 0.1
    n = small.n_vertices
    lhs = pr.log_density(d + dz[:n], beta + dz[n:]) - pr.log_density(d, beta)
    rhs = np.concatenate([gd, gb]) @ dz - 0.5 * dz @ (Q @ dz)
    assert lhs == pytest.approx(rhs, rel=1e-9)
