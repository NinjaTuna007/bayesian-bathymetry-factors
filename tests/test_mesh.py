import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M


@pytest.fixture
def m():
    return M.build(100.0, 130.0, 200.0, 220.0, h=1.5, buffer=2.0)


def plane(xy, a=0.3, b=-0.2, c=12.0):
    return a * xy[..., 0] + b * xy[..., 1] + c


def brute_locate(m, q):
    """Reference: barycentric weights of q in every triangle, pick the one with all >= 0."""
    p = m.vertices[m.triangles]
    T = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=-1)  # (m, 2, 2) columns
    lam = np.linalg.solve(T[None], (q[:, None, :] - p[None, :, 0])[..., None])[..., 0]
    bary = np.concatenate([1 - lam.sum(-1, keepdims=True), lam], axis=-1)
    hit = (bary >= -1e-9).all(-1)
    return hit, bary


def test_edges_and_areas(m):
    p = m.vertices[m.triangles]
    for k in range(3):
        np.testing.assert_allclose(np.linalg.norm(p[:, (k + 1) % 3] - p[:, k], axis=1), m.h)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    np.testing.assert_allclose(area, math.sqrt(3) / 4 * m.h**2)  # also checks CCW


def test_covers_box_plus_buffer(m):
    rng = np.random.default_rng(0)
    q = m.to_local(rng.uniform([98.0, 198.0], [132.0, 222.0], size=(5000, 2)))
    assert bool(M.locate(m, q)[2].all())


def test_locate_matches_brute_force(m):
    rng = np.random.default_rng(1)
    lo, hi = m.vertices.min(0) - 2, m.vertices.max(0) + 2
    q = rng.uniform(lo, hi, size=(2000, 2))
    tri, bary, inside = (np.asarray(a) for a in M.locate(m, q))
    hit, ref = brute_locate(m, q)
    np.testing.assert_array_equal(inside, hit.any(1))
    idx = np.flatnonzero(inside)
    assert hit[idx, tri[idx]].all()
    np.testing.assert_allclose(bary[idx], ref[idx, tri[idx]], atol=1e-9)


def test_interpolation_reproduces_vertices_and_planes(m):
    d = plane(m.vertices)
    np.testing.assert_allclose(M.interpolate(m, d, m.vertices), d, atol=1e-9)
    rng = np.random.default_rng(2)
    q = m.to_local(rng.uniform([100.0, 200.0], [130.0, 220.0], size=(500, 2)))
    np.testing.assert_allclose(M.interpolate(m, d, q), plane(q), atol=1e-9)
    dr = rng.standard_normal(m.n_vertices)
    np.testing.assert_allclose(M.interpolate(m, dr, m.vertices), dr, atol=1e-9)


def test_outside_is_nan(m):
    q = np.array([[1e4, 0.0], [0.0, -1e4]])
    assert np.isnan(M.interpolate(m, np.zeros(m.n_vertices), q)).all()
    assert np.isnan(M.normal(m, np.zeros(m.n_vertices), q)).all()


def test_face_gradient_and_normal_of_plane(m):
    d = plane(m.vertices)
    np.testing.assert_allclose(M.face_gradient(m, d), np.broadcast_to([0.3, -0.2], (len(m.triangles), 2)),
                               atol=1e-9)
    n = M.normal(m, d, np.zeros((1, 2)))[0]
    np.testing.assert_allclose(n, np.array([0.3, -0.2, 1.0]) / math.sqrt(1.13), atol=1e-9)


def test_gradients_wrt_depth_and_location(m):
    d = jnp.asarray(np.random.default_rng(3).standard_normal(m.n_vertices))
    q = jnp.array([[0.37, -1.21]])
    g = jax.grad(lambda d: M.interpolate(m, d, q).sum())(d)
    tri, bary, _ = M.locate(m, q)
    expect = np.zeros(m.n_vertices)
    expect[m.triangles[int(tri[0])]] = np.asarray(bary[0])
    np.testing.assert_allclose(g, expect, atol=1e-12)
    # :math:`d(q)` is linear inside a face, so :math:`\partial d / \partial q` is the face gradient
    gq = jax.grad(lambda q: M.interpolate(m, d, q).sum())(q)
    np.testing.assert_allclose(gq[0], M.face_gradient(m, d)[int(tri[0])], atol=1e-9)
