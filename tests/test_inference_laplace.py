import numpy as np
import pytest
import scipy.sparse as sp

from bbf import mesh as M
from bbf import prior as PR
from bbf.inference.laplace import marginal_variances, sample_precision


def spd(seed=0):
    """A sparse SPD matrix like H: a GMRF prior plus data rows on a few hundred vertices."""
    m = M.build(0.0, 12.0, 0.0, 10.0, h=1.0)
    rng = np.random.default_rng(seed)
    L = PR.precision(m, 3.0, 1.0)
    rows = sp.random(300, m.n_vertices, density=0.01, rng=seed)
    return sp.csc_matrix(L + 50.0 * rows.T @ rows + sp.diags(rng.uniform(0, 1, m.n_vertices)))


def test_samples_have_covariance_inverse_of_H():
    H = spd()
    n = H.get_shape()[0]
    X = sample_precision(H, 40000, rng=1)
    emp = X @ X.T / X.shape[1]
    ref = np.linalg.inv(H.toarray())
    i = np.argsort(-np.diag(ref))[:5]  # the most variable entries, where errors would show
    np.testing.assert_allclose(emp[np.ix_(i, i)], ref[np.ix_(i, i)], atol=0.03 * ref[i, i].max())
    assert X.shape == (n, 40000)


def test_rao_blackwell_variances():
    H = spd(2)
    ref = np.diag(np.linalg.inv(H.toarray()))
    var = marginal_variances(H, 200, rng=3)
    err = np.abs(var / ref - 1)
    # 200 samples: about 3% on average (so 1.5% in standard deviation), worst entries ~15%
    assert err.mean() < 0.05 and err.max() < 0.25
    # well ahead of the plain sample variance from the same samples (~2.5x here)
    plain = np.mean(sample_precision(H, 200, rng=3) ** 2, axis=1)
    assert err.mean() < 0.6 * np.mean(np.abs(plain / ref - 1))


def test_diagonal_precision_is_exact():
    H = sp.diags(np.array([1.0, 4.0, 0.25]))
    assert marginal_variances(H, 5, rng=0) == pytest.approx([1.0, 0.25, 4.0])
