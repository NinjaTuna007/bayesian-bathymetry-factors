r"""Sparse Jacobian :math:`J = \partial S / \partial d` of one ping-side, from sample-local derivatives.

Each curve sample :math:`k` adds :math:`w_k = a_k V_k \Delta q_k / \Delta r` to its two nearest bins.
With the sample locations :math:`q_k` held fixed, :math:`w_k` and its range depend on only six
vertex depths: the three of the triangle under :math:`q_k` (depth and face normal) and the three
under the crest sample casting its horizon (diffraction). With smooth normals the normal also
depends on the triangles around each corner, at most twelve vertices in all, plus the crest's three. Forward-mode derivatives with six
tangents per sample, scattered into the bins, give :math:`J` at about six renders' cost.

Holding :math:`q_k` fixed drops the along-track shift of the trace, which is zero for a level
ping and tiny under small pitch. :math:`J` only enters the Gauss--Newton curvature; gradients
come from the exact reverse mode of `bbf.inference.objective`.
"""
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from bbf import mesh as M
from bbf.ping import geometry as G
from bbf.ping.render import curve_length
from bbf.ping.theta import Theta


def _sample_weights(theta: Theta, pose, side: int, z_ref: float, dr: float, d6, q, bary, bary_c, q_c, grad_op,
                    dq, has, inside):
    r"""Bin weights :math:`(c_0, c_1)` of one sample from its six local vertex depths d6 (own 3, crest 3)."""
    s, _ = G.pose_frame(pose)
    dk, dc = d6[:3], d6[3:]
    P = jnp.concatenate([q, (z_ref - bary @ dk)[None]])
    g = grad_op @ jnp.stack([dk[1] - dk[0], dk[2] - dk[0]])
    n = jnp.concatenate([g, jnp.ones(1)])
    n = n / jnp.linalg.norm(n)
    a, r = G.echo(theta, pose, side, P[None], n[None])
    a, r = a[0], r[0]
    Pc = jnp.concatenate([q_c, (z_ref - bary_c @ dc)[None]])
    alpha = jnp.arctan2(P[2] - s[2], jnp.linalg.norm(P[:2] - s[:2]))
    alpha_c = jnp.arctan2(Pc[2] - s[2], jnp.linalg.norm(Pc[:2] - s[:2]))
    d1 = jnp.linalg.norm(Pc - s)
    d2 = jnp.maximum(jnp.linalg.norm(jnp.where(has, P - Pc, 1.0)), 1e-6)
    nu = d1 * jnp.sin(alpha_c - alpha) * jnp.sqrt(2 * (d1 + d2) / (theta.wavelength * d1 * d2))
    V = jnp.where(has, G.knife_edge(nu), 1.0)
    w = jnp.where(inside, a * V * dq, 0.0) / dr
    x = r / dr
    f = x - jax.lax.stop_gradient(jnp.floor(x))
    return jnp.stack([(1 - f) * w, f * w])


def _smooth_normal_local(dl, bary, corner, face, face_ok, face_grad):
    """Blended vertex normal of one sample from the depths dl of its triangle's neighbourhood."""
    def face_normal(idx, op):
        g = op @ jnp.stack([dl[idx[1]] - dl[idx[0]], dl[idx[2]] - dl[idx[0]]])
        n = jnp.concatenate([g, jnp.ones(1)])
        return n / jnp.linalg.norm(n)

    nf = jax.vmap(jax.vmap(face_normal))(face, face_grad)  # (3, 6, 3)
    vn = jnp.sum(jnp.where(face_ok[..., None], nf, 0.0), axis=1)
    vn = vn / jnp.linalg.norm(vn, axis=-1, keepdims=True)
    n = bary @ vn
    return n / jnp.linalg.norm(n), dl[corner]


def _weights_from(theta: Theta, pose, side: int, z_ref: float, dr: float, P, n, Pc, dq, has, inside):
    """Bin weights of one sample from its seabed point, normal and crest point."""
    s, _ = G.pose_frame(pose)
    a, r = G.echo(theta, pose, side, P[None], n[None])
    a, r = a[0], r[0]
    alpha = jnp.arctan2(P[2] - s[2], jnp.linalg.norm(P[:2] - s[:2]))
    alpha_c = jnp.arctan2(Pc[2] - s[2], jnp.linalg.norm(Pc[:2] - s[:2]))
    d1 = jnp.linalg.norm(Pc - s)
    d2 = jnp.maximum(jnp.linalg.norm(jnp.where(has, P - Pc, 1.0)), 1e-6)
    nu = d1 * jnp.sin(alpha_c - alpha) * jnp.sqrt(2 * (d1 + d2) / (theta.wavelength * d1 * d2))
    V = jnp.where(has, G.knife_edge(nu), 1.0)
    w = jnp.where(inside, a * V * dq, 0.0) / dr
    x = r / dr
    f = x - jax.lax.stop_gradient(jnp.floor(x))
    return jnp.stack([(1 - f) * w, f * w])


@cache
def _local_jacobian_smooth(theta: Theta, side: int, z_ref: float, dr: float):
    def one(pose, d15, q, bary, bary_c, q_c, corner, face, face_ok, face_grad, dq, has, inside):
        def f(d15):
            dl, dc = d15[:12], d15[12:]
            n, dk = _smooth_normal_local(dl, bary, corner, face, face_ok, face_grad)
            P = jnp.concatenate([q, (z_ref - bary @ dk)[None]])
            Pc = jnp.concatenate([q_c, (z_ref - bary_c @ dc)[None]])
            return _weights_from(theta, pose, side, z_ref, dr, P, n, Pc, dq, has, inside)
        return jax.jacfwd(f)(d15)
    return jax.jit(jax.vmap(one, in_axes=(None,) + (0,) * 12))


@cache
def _local_jacobian(theta: Theta, side: int, z_ref: float, dr: float):
    def one(pose, d6, q, bary, bary_c, q_c, grad_op, dq, has, inside):
        def f(d6):
            return _sample_weights(theta, pose, side, z_ref, dr, d6, q, bary, bary_c, q_c, grad_op, dq, has, inside)
        return jax.jacfwd(f)(d6)
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0)))


def ping_side_jacobian(mesh: M.Mesh, d, pose, side: int, theta: Theta, u, n_bins: int, dr: float) -> sp.csr_matrix:
    r""":math:`\partial S / \partial d` (n_bins, n_vertices) of one ping-side, sparse."""
    d = jnp.asarray(d)
    q = G.trace(mesh, d, pose, side, u)
    tri, bary, inside = (np.asarray(a) for a in M.locate(mesh, q))
    P, _, _ = G.samples(mesh, d, pose, side, u, smooth=theta.smooth_normals)
    s = np.asarray(G.pose_frame(pose)[0])
    Pn = np.asarray(P)
    alpha = np.arctan2(Pn[:, 2] - s[2], np.linalg.norm(Pn[:, :2] - s[:2], axis=-1))
    idx = np.searchsorted(np.asarray(u), np.asarray(u) - theta.shadow_margin, side="right") - 1
    has = idx >= 0
    at = np.asarray(G._running_max(jnp.asarray(alpha))[1])
    j = at[np.maximum(idx, 0)]
    inside = inside & (Pn[:, 2] <= theta.z_water)  # seabed above the water surface returns nothing
    verts = mesh.triangles[tri]
    dn = np.asarray(d)
    if theta.smooth_normals:
        tab = mesh.smooth_tables
        cols6 = np.concatenate([tab["ring"][tri], verts[j]], axis=1)  # (K, 15)
        jac = _local_jacobian_smooth(theta, side, mesh.z_ref, dr)(
            jnp.asarray(pose), jnp.asarray(dn[cols6]), q, jnp.asarray(bary), jnp.asarray(bary[j]), q[j],
            jnp.asarray(tab["corner"][tri]), jnp.asarray(tab["face"][tri]), jnp.asarray(tab["face_ok"][tri]),
            jnp.asarray(tab["face_grad"][tri]), curve_length(P[:, :2]), jnp.asarray(has), jnp.asarray(inside))
    else:
        cols6 = np.concatenate([verts, verts[j]], axis=1)  # (K, 6)
        jac = _local_jacobian(theta, side, mesh.z_ref, dr)(
            jnp.asarray(pose), jnp.asarray(dn[cols6]), q, jnp.asarray(bary), jnp.asarray(bary[j]), q[j],
            jnp.asarray(mesh._grad_ops[tri]), curve_length(P[:, :2]), jnp.asarray(has), jnp.asarray(inside))
    jac = np.asarray(jac)  # (K, 2, n_local)
    nl = cols6.shape[1]
    r = np.linalg.norm(Pn - s, axis=-1)
    i0 = np.floor(r / dr).astype(int)
    rows = np.stack([i0, i0 + 1], axis=1)[:, :, None].repeat(nl, axis=2)
    cols = cols6[:, None, :].repeat(2, axis=1)
    keep = (rows >= 0) & (rows < n_bins) & (jac != 0)
    return sp.csr_matrix(sp.coo_matrix((jac[keep], (rows[keep], cols[keep])), shape=(n_bins, mesh.n_vertices)))
