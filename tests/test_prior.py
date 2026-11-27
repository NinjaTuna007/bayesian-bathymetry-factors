import math

import numpy as np
import pytest
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


def make_prior(m, rho, sigma, alpha=2):
    return P.Prior(m, rho, sigma, mu_beta=np.zeros(3), Sigma_beta=np.eye(3), alpha=alpha)


def covariance_column(prior, i):
    from sksparse.cholmod import cho_factor
    e = np.zeros(prior.mesh.n_vertices)
    e[i] = 1.0
    return np.asarray(cho_factor(prior.Lambda).solve(e))


def nearest(m, xy):
    return int(np.argmin(np.linalg.norm(m.vertices - xy, axis=1)))


@pytest.mark.parametrize("alpha", [2, 3])
def test_matern_variance_and_range(alpha):
    rho, sigma = 10.0, 0.5
    m = M.build(-50.0, 50.0, -50.0, 50.0, h=rho / 10, origin=(0.0, 0.0))
    pr = make_prior(m, rho, sigma, alpha)
    i = nearest(m, (0.0, 0.0))
    cov = covariance_column(pr, i)
    assert cov[i] == pytest.approx(sigma**2, rel=0.05)
    nu = alpha - 1
    k = math.sqrt(8 * nu) / rho

    def corr(r):
        r"""Matérn correlation :math:`(\kappa r)^\nu K_\nu(\kappa r) / (2^{\nu - 1} \Gamma(\nu))`."""
        return (k * r) ** nu * kv(nu, k * r) / (2 ** (nu - 1) * math.gamma(nu))

    for xy in [(rho, 0.0), (0.0, rho), (-rho * 0.6, -rho * 0.8)]:
        j = nearest(m, xy)
        r = np.linalg.norm(m.vertices[j] - m.vertices[i])
        assert cov[j] / cov[i] == pytest.approx(corr(r), abs=0.02)
    assert corr(rho) == pytest.approx(0.14, abs=0.01)


def test_slope_variance_depends_on_the_mesh_only_for_alpha_2():
    r"""Prior variance of a face gradient: grows as :math:`h` shrinks for :math:`\alpha = 2` (:math:`\nu = 1`), settles for :math:`\alpha = 3`."""
    rho, sigma = 10.0, 1.0

    def slope_var(h, alpha):
        m = M.build(-40.0, 40.0, -40.0, 40.0, h=h, origin=(0.0, 0.0))
        pr = make_prior(m, rho, sigma, alpha)
        c = np.asarray(M.locate(m, np.zeros((1, 2)))[0])[0]
        t = m.triangles[c]
        G = m._grad_ops[c] @ np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])  # (2, 3): face gradient of the corner depths
        cols = np.stack([covariance_column(pr, i) for i in t], axis=1)[t]  # (3, 3) covariance of the corners
        return float(np.trace(G @ cols @ G.T))

    r2 = slope_var(0.5, 2) / slope_var(1.0, 2)
    r3 = slope_var(0.5, 3) / slope_var(1.0, 3)
    assert r2 > 1.1 and abs(r3 - 1) < 0.08, (r2, r3)


def test_constant_mean_prior(small):
    pr = P.Prior(small, 3.0, 0.5, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]), alpha=3)
    assert pr.X.shape == (small.n_vertices, 1) and pr.n_beta == 1
    rng = np.random.default_rng(5)
    d, beta = rng.standard_normal(small.n_vertices) + 15, np.array([14.0])
    _, gb = pr.grad(d, beta)
    eps = 1e-6
    fd = (pr.log_density(d, beta + eps) - pr.log_density(d, beta - eps)) / (2 * eps)
    assert gb[0] == pytest.approx(fd, rel=1e-6)
    assert pr.joint_precision().shape == (small.n_vertices + 1,) * 2
    u, b = P.sample(pr, rng)
    assert u.shape == (small.n_vertices,) and b.shape == (1,)


def test_boundary_inflation_and_buffer():
    rho, sigma = 10.0, 0.5
    m = M.build(-30.0, 30.0, -30.0, 30.0, h=rho / 10, origin=(0.0, 0.0))
    pr = make_prior(m, rho, sigma)
    edge = nearest(m, (0.0, m.vertices[:, 1].min()))
    inner = nearest(m, (0.0, m.vertices[:, 1].min() + 2 * rho))
    assert covariance_column(pr, edge)[edge] > 1.5 * sigma**2
    assert covariance_column(pr, inner)[inner] == pytest.approx(sigma**2, rel=0.1)


@pytest.mark.parametrize("alpha", [2, 3])
def test_sample_covariance(alpha):
    rho, sigma = 4.0, 0.5
    m = M.build(-10.0, 10.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    pr = make_prior(m, rho, sigma, alpha)
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


def test_slope_factor_is_the_sum_of_squared_face_gradients():
    m = M.build(0.0, 8.0, 0.0, 7.0, h=0.5)
    s = 0.2
    base = P.Prior(m, 3.0, 2.0, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]))
    pr = P.Prior(m, 3.0, 2.0, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]), slope=s)
    rng = np.random.default_rng(0)
    d, beta = 15.0 + rng.standard_normal(m.n_vertices), np.array([15.0])
    g = np.asarray(M.face_gradient(m, d))
    assert pr.log_density(d, beta) - base.log_density(d, beta) == pytest.approx(-0.5 * np.sum(g**2) / s**2, rel=1e-9)


def test_slope_factor_limits_the_tilt_of_prior_draws():
    m = M.build(0.0, 30.0, 0.0, 30.0, h=1.0)
    tilt = {}
    for s in (None, 0.2):
        pr = P.Prior(m, 5.0, 8.0, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]), slope=s)
        d, _ = P.sample(pr, rng=1)
        tilt[s] = np.sqrt(np.mean(np.asarray(M.face_gradient(m, d)) ** 2))
    assert tilt[None] > 1.0  # the Matérn part alone: slopes of order one at the mesh scale
    assert tilt[0.2] < 0.2


def _student(nu=1.0, s=0.2):
    m = M.build(0.0, 8.0, 0.0, 7.0, h=0.5)
    pr = P.Prior(m, 3.0, 2.0, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]), slope=s, slope_nu=nu)
    d = 15.0 + np.random.default_rng(3).standard_normal(m.n_vertices)
    return m, pr, d, np.array([15.2])


def test_student_slope_density_and_gradient():
    m, pr, d, beta = _student()
    g = np.asarray(M.face_gradient(m, d))
    base = P.Prior(m, 3.0, 2.0, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]))
    want = -1.5 * np.sum(np.log1p(np.sum(g**2, axis=1) / (3 * 0.04)))
    assert pr.log_density(d, beta) - base.log_density(d, beta) == pytest.approx(want, rel=1e-9)
    gd, gb = pr.grad(d, beta)
    e = np.zeros_like(d)
    for i in (0, 17, 101):
        e[:] = 0.0
        e[i] = 1e-6
        fd = (pr.log_density(d + e, beta) - pr.log_density(d - e, beta)) / 2e-6
        assert gd[i] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    fb = (pr.log_density(d, beta + 1e-6) - pr.log_density(d, beta - 1e-6)) / 2e-6
    assert gb[0] == pytest.approx(fb, rel=1e-5)


def test_student_slope_reweighted_curvature():
    m, pr, d, beta = _student()
    # reweighted at the point, its precision gives the exact gradient there
    np.testing.assert_allclose(pr.at(d, beta).Lambda @ (d - pr.X @ beta), -pr.grad(d, beta)[0], rtol=1e-9, atol=1e-9)
    # at a level seabed it is the Gaussian factor's, so the start does not change
    gauss = P.Prior(m, 3.0, 2.0, mu_beta=np.array([15.0]), Sigma_beta=np.array([[4.0]]), slope=0.2)
    np.testing.assert_allclose(pr.Lambda.toarray(), gauss.Lambda.toarray(), rtol=1e-9, atol=1e-9)


def test_student_slope_makes_a_bank_cheaper():
    m = M.build(0.0, 20.0, 0.0, 10.0, h=1.0)
    x = m.vertices[:, 0]
    bank = np.clip(x - 8.0, 0.0, 8.0) * np.tan(np.radians(40.0))  # rises 6.7 m over 8 m
    cost = {}
    for nu in (None, 1.0):
        pr = P.Prior(m, 5.0, 8.0, mu_beta=np.array([0.0]), Sigma_beta=np.array([[4.0]]), slope=0.2, slope_nu=nu)
        cost[nu] = pr.log_density(np.zeros(m.n_vertices), np.zeros(1)) - pr.log_density(bank, np.zeros(1))
    assert cost[1.0] < 0.5 * cost[None]


def test_bowl_design_rises_to_a_rim_and_enters_the_prior():
    from bbf import mesh as M
    from bbf.prior import Prior, bowl_design

    m = M.build(0.0, 40.0, 0.0, 10.0, h=1.0)
    x0 = float(np.median(m.vertices[:, 0])) - 15.0
    tracks = np.column_stack([np.full(401, x0), np.linspace(m.vertices[:, 1].min() - 5, m.vertices[:, 1].max() + 5, 401)])  # along x = x0
    X = bowl_design(m, tracks, scale=20.0)
    assert X.shape == (m.n_vertices, 2)
    s_true = np.abs(m.vertices[:, 0] - x0)
    np.testing.assert_allclose(X[:, 0], 1.0)
    np.testing.assert_allclose(X[:, 1], -np.minimum(s_true, 20.0) ** 2 / 400.0, atol=1e-3)
    on, far = s_true < 0.3, s_true > 20.0
    assert on.any() and far.any()
    pr = Prior(m, 5.0, 2.0, mu_beta=np.array([16.0, 0.0]), Sigma_beta=np.diag([100.0, 100.0]), design=X)
    beta = np.array([16.0, 8.0])
    d = X @ beta
    assert d[on].mean() > 15.9 and abs(d[far].mean() - 8.0) < 1e-9
    # the bowl itself is the trend: it costs only the coefficient's prior
    np.testing.assert_allclose(pr.log_density(d, beta), pr.log_density(X @ pr.mu_beta, pr.mu_beta) - 0.5 * 8.0**2 / 100.0)
    assert pr.joint_precision().shape == (m.n_vertices + 2, m.n_vertices + 2)


def test_slope_faces_switch_the_factor_per_face():
    from bbf import mesh as M
    from bbf.prior import Prior

    m = M.build(0.0, 12.0, 0.0, 8.0, h=1.0)
    rng = np.random.default_rng(3)
    d = 10.0 + rng.normal(0, 0.3, m.n_vertices)
    beta = np.array([10.0])
    kw = {"mu_beta": np.array([10.0]), "Sigma_beta": np.array([[100.0]])}
    nf = len(m.triangles)
    for nu in (None, 1.0):
        full = Prior(m, 5.0, 8.0, slope=0.2, slope_nu=nu, **kw)
        ones = Prior(m, 5.0, 8.0, slope=0.2, slope_nu=nu, slope_faces=np.ones(nf, bool), **kw)
        none = Prior(m, 5.0, 8.0, slope=0.2, slope_nu=nu, slope_faces=np.zeros(nf, bool), **kw)
        plain = Prior(m, 5.0, 8.0, **kw)
        np.testing.assert_allclose(ones.log_density(d, beta), full.log_density(d, beta), rtol=1e-10)  # all on: the factor as before
        np.testing.assert_allclose(ones.Lambda.toarray(), full.Lambda.toarray(), atol=1e-8)
        np.testing.assert_allclose(none.log_density(d, beta), plain.log_density(d, beta), rtol=1e-10)  # all off: the Matern part alone
        np.testing.assert_allclose(none.at(d, beta).Lambda.toarray(), plain.Lambda.toarray(), atol=1e-10)
        half = np.zeros(nf, bool)
        half[: nf // 2] = True
        pr = Prior(m, 5.0, 8.0, slope=0.2, slope_nu=nu, slope_faces=half, **kw)
        g = pr.grad(d, beta)[0]
        eps, num = 1e-6, np.zeros(m.n_vertices)
        for i in range(0, m.n_vertices, 7):
            e = np.zeros(m.n_vertices)
            e[i] = eps
            num[i] = (pr.log_density(d + e, beta) - pr.log_density(d - e, beta)) / (2 * eps)
        np.testing.assert_allclose(g[::7], num[::7], rtol=1e-5, atol=1e-6)
        assert plain.log_density(d, beta) > pr.log_density(d, beta) > full.log_density(d, beta)
        assert pr.at(d, beta).slope_faces is half  # carried through the reweighting
