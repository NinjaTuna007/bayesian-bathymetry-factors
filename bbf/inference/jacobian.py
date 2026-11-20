r"""Sparse Jacobian :math:`J = \partial S / \partial d` of one ping-side, from sample-local derivatives.

Each curve sample :math:`k` adds :math:`w_k = a_k V_k \Delta q_k / \Delta r` to its two nearest bins.
With the sample locations :math:`q_k` held fixed, :math:`w_k` and its range depend on only six
vertex depths: the three of the triangle under :math:`q_k` (depth and face normal) and the three
under the crest sample casting its horizon (diffraction). Forward-mode derivatives with six
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
    P, _, _ = G.samples(mesh, d, pose, side, u)
    s = np.asarray(G.pose_frame(pose)[0])
    Pn = np.asarray(P)
    alpha = np.arctan2(Pn[:, 2] - s[2], np.linalg.norm(Pn[:, :2] - s[:2], axis=-1))
    idx = np.searchsorted(np.asarray(u), np.asarray(u) - theta.shadow_margin, side="right") - 1
    has = idx >= 0
    at = np.asarray(G._running_max(jnp.asarray(alpha))[1])
    j = at[np.maximum(idx, 0)]
    verts = mesh.triangles[tri]
    v6 = np.concatenate([verts, verts[j]], axis=1)
    dn = np.asarray(d)
    jac = _local_jacobian(theta, side, mesh.z_ref, dr)(
        jnp.asarray(pose), jnp.asarray(dn[v6]), q, jnp.asarray(bary), jnp.asarray(bary[j]), q[j],
        jnp.asarray(mesh._grad_ops[tri]), curve_length(P[:, :2]), jnp.asarray(has), jnp.asarray(inside))
    jac = np.asarray(jac)  # (K, 2, 6)
    r = np.linalg.norm(Pn - s, axis=-1)
    i0 = np.floor(r / dr).astype(int)
    rows = np.stack([i0, i0 + 1], axis=1)[:, :, None].repeat(6, axis=2)
    cols = v6[:, None, :].repeat(2, axis=1)
    keep = (rows >= 0) & (rows < n_bins) & (jac != 0)
    return sp.csr_matrix(sp.coo_matrix((jac[keep], (rows[keep], cols[keep])), shape=(n_bins, mesh.n_vertices)))
