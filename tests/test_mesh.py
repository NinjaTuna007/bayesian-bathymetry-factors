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


def interior_edges(m):
    """Edges shared by two triangles: (vertex pair, triangle pair) per edge."""
    owners = {}
    for k, t in enumerate(m.triangles):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            owners.setdefault((min(a, b), max(a, b)), []).append(k)
    return [(e, ts) for e, ts in owners.items() if len(ts) == 2]


def test_gradients_across_triangle_edges(m):
    r"""Across a shared edge, :math:`d(q)` and :math:`\partial d / \partial d_i = \psi_i(q)` are
    continuous, while :math:`\partial d / \partial q` jumps between the two face gradients.

    The ping factor differentiates through all three, so each side of the edge, and the edge
    itself, must give the gradient of a triangle that actually touches the point.
    """
    rng = np.random.default_rng(5)
    d = jnp.asarray(rng.standard_normal(m.n_vertices))
    edges = interior_edges(m)
    picked = [edges[k] for k in rng.choice(len(edges), size=60, replace=False)]
    v = m.vertices
    directions = {round(math.degrees(math.atan2(*(v[j] - v[i])[::-1])) % 180) for (i, j), _ in picked}
    assert directions == {0, 60, 120}  # all three edge orientations of the lattice

    grad_q = jax.grad(lambda q: M.interpolate(m, d, q).sum())
    grad_d = jax.grad(lambda d, q: M.interpolate(m, d, q).sum())
    fg = np.asarray(M.face_gradient(m, d))
    eps = 1e-6
    for (i, j), (t1, t2) in picked:
        mid = (v[i] + v[j]) / 2
        e = (v[j] - v[i]) / m.h
        nrm = np.array([-e[1], e[0]])
        sides = mid + np.array([eps, -eps])[:, None] * nrm
        # depth is continuous: both sides agree with the edge value to first order in eps
        on_edge = float(M.interpolate(m, d, mid[None])[0])
        assert on_edge == pytest.approx(float((d[i] + d[j]) / 2), abs=1e-9)
        np.testing.assert_allclose(M.interpolate(m, d, sides), on_edge, atol=10 * eps)
        # the hat-function weights are continuous: half on each edge vertex, from either side
        for q in (mid, *sides):
            g = np.asarray(grad_d(d, jnp.asarray(q[None])))
            expect = np.zeros(m.n_vertices)
            expect[[i, j]] = 0.5
            np.testing.assert_allclose(g, expect, atol=1e-5)
        # the spatial gradient is one-sided: each side gives its own face's gradient
        tri_sides = {int(t) for t in np.asarray(M.locate(m, sides)[0])}
        assert tri_sides == {t1, t2}
        for q in sides:
            t = int(M.locate(m, q[None])[0][0])
            np.testing.assert_allclose(grad_q(jnp.asarray(q[None]))[0], fg[t], atol=1e-9)
        # on the edge itself it is finite and belongs to one of the two faces
        g_mid = np.asarray(grad_q(jnp.asarray(mid[None]))[0])
        assert np.isfinite(g_mid).all()
        assert min(np.abs(g_mid - fg[t1]).max(), np.abs(g_mid - fg[t2]).max()) < 1e-9


def test_seabed_point_uses_z_ref():
    m = M.build(0.0, 10.0, 0.0, 10.0, h=1.0, z_ref=55.0)
    d = plane(m.vertices)
    q = np.array([[0.3, -1.2], [2.0, 1.0]])
    P = np.asarray(M.seabed_point(m, d, q))
    np.testing.assert_allclose(P[:, :2], q)
    np.testing.assert_allclose(P[:, 2], 55.0 - plane(q), atol=1e-9)


def test_smooth_normals_equal_face_normals_on_a_plane(m):
    d = plane(m.vertices)
    q = m.to_local(np.random.default_rng(7).uniform([100.0, 200.0], [130.0, 220.0], size=(200, 2)))
    np.testing.assert_allclose(M.smooth_normal(m, d, q), M.normal(m, d, q), atol=1e-12)


def test_smooth_normals_are_continuous_across_edges(m):
    d = jnp.asarray(np.random.default_rng(8).standard_normal(m.n_vertices) * 0.5)
    (i, j), _ = interior_edges(m)[len(interior_edges(m)) // 2]
    v = m.vertices
    mid = (v[i] + v[j]) / 2
    e = (v[j] - v[i]) / m.h
    nrm = np.array([-e[1], e[0]])
    eps = 1e-6
    sides = np.stack([mid + eps * nrm, mid - eps * nrm])
    jump_face = np.linalg.norm(np.diff(np.asarray(M.normal(m, d, sides)), axis=0))
    jump_smooth = np.linalg.norm(np.diff(np.asarray(M.smooth_normal(m, d, sides)), axis=0))
    assert jump_face > 1e-3 and jump_smooth < 1e-4
