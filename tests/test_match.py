import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, polygamma

from bbf import match as Mt
from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 400, 0.1
LEVEL = jnp.array([0.0, 0.3, SONAR_Z, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def scene():
    m = M.build(-45.0, 45.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    d = jnp.asarray(DEPTH - 1.5 * np.sin(v[:, 0] / 6) * np.cos(v[:, 1] / 5))
    xi = jnp.asarray(0.5 * np.cos(v[:, 0] / 4) * np.sin(v[:, 1] / 3 + 1.0))
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)
    return m, d, xi, th, P.u_grid(N_BINS, DR, du=0.02)


@pytest.mark.parametrize("ka,kb", [(0.48, 0.48), (3.0, 11.0), (40.0, 7.5)])
def test_log_ratio_pdf_normalised_and_matches_simulation(ka, kb):
    x = np.linspace(-60, 60, 400001)
    p = np.exp(Mt.log_ratio_pdf(x, ka, kb))
    assert np.trapezoid(p, x) == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(0)
    e = np.log(rng.gamma(kb, 1 / kb, 400000)) - np.log(rng.gamma(ka, 1 / ka, 400000))
    mean = digamma(kb) - np.log(kb) - digamma(ka) + np.log(ka)
    var = polygamma(1, ka) + polygamma(1, kb)
    assert np.trapezoid(x * p, x) == pytest.approx(mean, abs=1e-6)
    assert e.mean() == pytest.approx(mean, abs=4 * np.sqrt(var / e.size))
    assert e.var() == pytest.approx(var, rel=0.02)
    # the density itself, against the simulated histogram around the mode
    hist, edges = np.histogram(e, bins=20, range=(mean - 1.5 * np.sqrt(var), mean + 1.5 * np.sqrt(var)))
    mid = 0.5 * (edges[1:] + edges[:-1])
    np.testing.assert_allclose(hist / (e.size * np.diff(edges)), np.exp(Mt.log_ratio_pdf(mid, ka, kb)), rtol=0.05)


def test_cell_scores_of_simulated_passes_are_standard_normal():
    r"""Cells with their own reflectivity, bins of unequal means: :math:`z` is :math:`\mathcal{N}(0, 1)`; a model error shows."""
    rng = np.random.default_rng(1)
    L, n_cells = 0.6, 4000
    cells = []
    kappa = np.exp(rng.normal(0, 1.5, n_cells))
    for n_bins in (12, 30):  # the two passes see each cell with different numbers of bins
        cell = np.repeat(np.arange(n_cells), n_bins)
        q = np.column_stack([cell + rng.uniform(0.05, 0.95, cell.size), rng.uniform(0.05, 0.95, cell.size)])
        J = np.exp(rng.uniform(-1, 1, cell.size))  # prediction per unit reflectivity
        y = rng.gamma(L, kappa[cell] * J / L)
        use = rng.random(cell.size) < 0.9
        cells.append((Mt.cell_sums(q, y, J, use, 1.0, L), q, y, J, use))
    a, b = cells[0][0], cells[1][0]
    assert len(a.key) == n_cells and np.all(a.shape <= 12 * L) and np.all(b.shape > a.shape.min())
    key, e, ka, kb = Mt.cell_residual(a, b)
    assert len(e) == n_cells and key.shape == (n_cells, 2)
    z = Mt.ratio_score(e, ka, kb)
    assert z.mean() == pytest.approx(0.0, abs=0.06) and z.std() == pytest.approx(1.0, abs=0.05)
    # pass b twice as bright as predicted: the scores shift
    _, q, y, J, use = cells[1]
    _, e2, ka, kb = Mt.cell_residual(a, Mt.cell_sums(q, 2.0 * y, J, use, 1.0, L))
    assert np.mean(Mt.ratio_score(e2, ka, kb)) > 1.0
    # far residuals keep their size and sign
    far = Mt.ratio_score(np.array([-5.0, 5.0]), 50.0, 50.0)
    assert far[0] < -12 and far[1] > 12 and np.all(np.isfinite(far))


def test_cell_residual_takes_common_cells_only():
    q = np.array([[0.5, 0.5], [0.6, 0.4], [1.5, 0.5], [np.nan, 0.0]])
    a = Mt.cell_sums(q, np.array([1.0, 3.0, 5.0, 7.0]), np.array([2.0, 2.0, 5.0, 1.0]), np.ones(4, bool), 1.0, 2.0)
    np.testing.assert_array_equal(a.key, [[0, 0], [1, 0]])
    np.testing.assert_allclose(a.y, [4.0, 5.0])
    np.testing.assert_allclose(a.shape, [4.0, 2.0])  # two equal bins double the shape
    b = Mt.cell_sums(q[2:3] + [0.0, 3.0], np.array([1.0]), np.array([1.0]), np.ones(1, bool), 1.0, 2.0)
    assert len(Mt.cell_residual(a, b)[1]) == 0
    c = Mt.cell_sums(q[2:3], np.array([10.0]), np.array([5.0]), np.ones(1, bool), 1.0, 2.0)
    key, e, _, _ = Mt.cell_residual(a, c)
    np.testing.assert_array_equal(key, [[1, 0]])
    np.testing.assert_allclose(e, [np.log(2.0)])


@pytest.mark.parametrize("side", [0, 1])
def test_field_scales_the_render(scene, side):
    r"""A constant :math:`\xi` multiplies :math:`S` by :math:`e^\xi`; a varying one weights each sample by its own ground."""
    m, d, xi, th, u = scene
    S0 = np.asarray(P.unit_intensity(m, d, LEVEL, side, th, u, N_BINS, DR))
    Sc = np.asarray(P.unit_intensity(m, d, LEVEL, side, th, u, N_BINS, DR, jnp.full(m.n_vertices, 0.7)))
    np.testing.assert_allclose(Sc, np.exp(0.7) * S0, rtol=1e-12)
    w, r, q = P.sample_weights(m, d, LEVEL, side, th, u, DR)
    wx, _, _ = P.sample_weights(m, d, LEVEL, side, th, u, DR, xi)
    np.testing.assert_allclose(wx, w * np.exp(np.asarray(M.interpolate(m, xi, q))), rtol=1e-12)
    Sx = np.asarray(P.unit_intensity(m, d, LEVEL, side, th, u, N_BINS, DR, xi))
    np.testing.assert_allclose(Sx, np.asarray(P.split_into_bins(r, wx, N_BINS, DR)), rtol=1e-12)
    lit = S0 > 1e-3 * S0.max()
    ratio = np.log(Sx[lit] / S0[lit])
    assert ratio.min() < -0.2 and ratio.max() > 0.2 and np.abs(ratio).max() <= 0.5 + 1e-9


def test_field_gradient_matches_finite_differences(scene):
    m, d, xi, th, u = scene
    wgt = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, N_BINS))

    def f(x):
        return jnp.sum(wgt * jnp.log(P.unit_intensity(m, d, LEVEL, 1, th, u, N_BINS, DR, x) + 1e-9))

    g = np.asarray(jax.grad(f)(xi))
    v = m.vertices
    for k in range(3):
        dx = jnp.asarray(np.sin(v[:, 0] / (3 + k)) * np.cos(v[:, 1] / 4 + k))
        h = 1e-6
        fd = float(f(xi + h * dx) - f(xi - h * dx)) / (2 * h)
        assert g @ np.asarray(dx) == pytest.approx(fd, rel=1e-6)


def test_footprints_follow_the_ground(scene):
    """On a level seabed the footprint of bin b lies outboard at the ground range of its slant range."""
    m, _, _, th, u = scene
    d = jnp.full(m.n_vertices, DEPTH)
    fp = Mt.footprints(m, d, np.asarray(LEVEL)[None], th, u, N_BINS, DR)
    assert fp.shape == (1, 2, N_BINS, 2)
    alt = DEPTH + SONAR_Z
    b = np.arange(N_BINS)
    echo = b * DR > alt + 1.0
    ground = np.sqrt(np.maximum((b * DR) ** 2 - alt**2, 0.0))
    assert np.isnan(fp[0, :, b * DR < alt - 2 * DR]).all()  # water column
    np.testing.assert_allclose(fp[0, 1, echo, 0], ground[echo], atol=0.1)  # starboard is +x at zero yaw
    np.testing.assert_allclose(fp[0, 0, echo, 0], -ground[echo], atol=0.1)
    np.testing.assert_allclose(fp[0, :, echo, 1], 0.3, atol=1e-9)


def test_reflectivity_prior_is_the_matern_field(scene):
    m = scene[0]
    refl = Mt.Reflectivity(m, 4.0, 0.7)
    np.testing.assert_allclose(refl.Lambda.toarray(), PR.precision(m, 4.0, 0.7).toarray())
    x = np.asarray(scene[2])
    h = 1e-6
    dx = np.cos(m.vertices[:, 0] / 3)
    fd = (refl.log_density(x + h * dx) - refl.log_density(x - h * dx)) / (2 * h)
    assert refl.grad(x) @ dx == pytest.approx(fd, rel=1e-6)
    assert refl.log_density(np.zeros_like(x)) == 0.0 and refl.log_density(x) < 0
