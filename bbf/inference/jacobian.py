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
from bbf.ping.theta import Theta, psi_of, render_key, render_theta


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
def _batched(mesh: M.Mesh, key: tuple, side: int, u_key: tuple, dr: float):
    r"""Jitted (d, poses, :math:`\psi`) -> per-ping local Jacobians (P, K, 2, 6), first bins (P, K), columns (P, K, 6).

    Keyed on the mesh and the `render_key` of theta, so it compiles once however theta's noise
    model and :math:`\psi` change between rounds.
    """
    u = np.asarray(u_key)
    _, _, margin, z_water = key
    idx = np.searchsorted(u, u - margin, side="right") - 1
    has = jnp.asarray(idx >= 0)
    idx0 = jnp.asarray(np.maximum(idx, 0))
    tris = jnp.asarray(mesh.triangles)
    grad_ops = jnp.asarray(mesh._grad_ops)

    def one(d, pose, psi):
        theta = render_theta(key, psi)
        s = G.pose_frame(pose)[0]
        q = G.trace(mesh, d, pose, side, jnp.asarray(u))
        tri, bary, inside = M.locate(mesh, q)
        verts = tris[tri]
        P = jnp.concatenate([q, (mesh.z_ref - jnp.sum(d[verts] * bary, axis=-1))[:, None]], axis=-1)
        alpha = G.elevation(P, s, inside)
        j = G._running_max(alpha)[1][idx0]
        inside = inside & (P[:, 2] <= z_water)  # seabed above the water surface returns nothing
        dq = curve_length(P[:, :2])
        cols = jnp.concatenate([verts, verts[j]], axis=1)

        def f(d6, q, bary, bary_c, q_c, grad_op, dq, has, inside):
            return _sample_weights(theta, pose, side, mesh.z_ref, dr, d6, q, bary, bary_c, q_c, grad_op, dq, has,
                                   inside)

        jac = jax.vmap(jax.jacfwd(f))(d[cols], q, bary, bary[j], q[j], grad_ops[tri], dq, has, inside)
        i0 = jnp.floor(jnp.linalg.norm(P - s, axis=-1) / dr).astype(jnp.int32)
        return jac, i0, cols

    return jax.jit(jax.vmap(one, in_axes=(None, 0, None)))


def ping_side_jacobians(mesh: M.Mesh, d, poses, side: int, theta: Theta, u, n_bins: int, dr: float,
                        chunk: int = 32) -> list[sp.csr_matrix]:
    r""":math:`\partial S / \partial d` (n_bins, n_vertices) of one side of every pose in poses (n, 6), sparse.

    Pings are processed `chunk` at a time in one compiled call; only the sparse assembly is in numpy.
    """
    from bbf.inference.objective import _chunks
    f = _batched(mesh, render_key(theta), side, tuple(np.asarray(u).tolist()), dr)
    d, psi = jnp.asarray(d), jnp.asarray(psi_of(theta))
    poses = np.asarray(poses)
    out = []
    for idx, padded in _chunks(len(poses), chunk):
        jac, i0, cols = (np.asarray(a) for a in f(d, jnp.asarray(poses[padded]), psi))
        nl = cols.shape[-1]
        for k in range(len(idx)):
            rows = np.stack([i0[k], i0[k] + 1], axis=1)[:, :, None].repeat(nl, axis=2)
            cc = cols[k][:, None, :].repeat(2, axis=1)
            keep = (rows >= 0) & (rows < n_bins) & (jac[k] != 0)
            out.append(sp.csr_matrix(sp.coo_matrix((jac[k][keep], (rows[keep], cc[keep])),
                                                   shape=(n_bins, mesh.n_vertices))))
    return out


def ping_side_jacobian(mesh: M.Mesh, d, pose, side: int, theta: Theta, u, n_bins: int, dr: float) -> sp.csr_matrix:
    r""":math:`\partial S / \partial d` (n_bins, n_vertices) of one ping-side, sparse."""
    return ping_side_jacobians(mesh, d, np.asarray(pose)[None], side, theta, u, n_bins, dr, chunk=1)[0]
