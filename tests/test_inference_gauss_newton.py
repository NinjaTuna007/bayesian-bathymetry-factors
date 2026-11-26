import numpy as np


def test_coarse_basis_is_a_partition_of_unity_and_reproduces_planes():
    from bbf import mesh as M
    from bbf.inference.gauss_newton import coarse_basis

    m = M.build(0.0, 30.0, 0.0, 20.0, h=1.0)
    P = coarse_basis(m, 8.0)
    assert P.shape[0] == m.n_vertices and P.shape[1] < m.n_vertices / 20
    np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)
    assert P.min() >= 0.0
    # nodal values of a plane give the plane at the vertices: find the nodes by least squares
    plane = 0.3 * m.vertices[:, 0] - 0.2 * m.vertices[:, 1] + 5.0
    c = np.linalg.lstsq(P.toarray(), plane, rcond=None)[0]
    np.testing.assert_allclose(P @ c, plane, atol=1e-8)
