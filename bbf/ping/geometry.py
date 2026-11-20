r"""Ping geometry: pose, the seabed curve a ping reaches, and the echo it returns.

World frame: local east :math:`x` and north :math:`y` of the mesh, world height :math:`z`
(up). Sonar frame: forward, port, up. A pose is the 6-vector
:math:`(x, y, z, \phi_r, \phi_p, \psi)` of position and roll, pitch, yaw as the navigation
gives them (NED/FRD, yaw clockwise from north).

Each side :math:`c` has a horizontal outboard axis and a down axis,
:math:`u_\mathrm{out} = R R_c e_y` and :math:`u_\mathrm{dwn} = R R_c e_z`, from the mounting
rotations `R_MOUNT`. The beam angle of a ray :math:`e` is
:math:`\beta_c = \operatorname{atan2}(u_\mathrm{dwn}^\top e,\, u_\mathrm{out}^\top e)`, as in the
paper with :math:`\pm u_\mathrm{stb}` folded into :math:`u_\mathrm{out}`.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.ping.theta import PORT, Theta

# Proper rotations in the sonar frame whose :math:`e_y` points outboard and :math:`e_z` down.
R_MOUNT = np.stack([np.diag([-1.0, 1, -1]),   # port: 180° about the port axis
                    np.diag([1.0, -1, -1])])  # starboard: 180° about the forward axis
_NED_TO_ENU = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
_FLU_TO_FRD = np.diag([1.0, -1, -1])


def pose_frame(pose):
    r"""Sonar position :math:`s` (3,) and rotation :math:`R` (3, 3), sonar to world.

    :math:`R = T_\mathrm{ENU \leftarrow NED}\, R_z(\psi) R_y(\phi_p) R_x(\phi_r)\, T_\mathrm{FRD \leftarrow FLU}`.
    """
    pose = jnp.asarray(pose)
    cr, sr = jnp.cos(pose[3]), jnp.sin(pose[3])
    cp, sp = jnp.cos(pose[4]), jnp.sin(pose[4])
    cy, sy = jnp.cos(pose[5]), jnp.sin(pose[5])
    r_ned = jnp.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return pose[:3], jnp.asarray(_NED_TO_ENU) @ r_ned @ jnp.asarray(_FLU_TO_FRD)


def _depth(mesh: M.Mesh, d, q):
    """Depth at q and whether q is on the mesh; finite everywhere (extrapolated off the mesh)."""
    tri, bary, inside = M.locate(mesh, q)
    dv = jnp.asarray(d)[jnp.asarray(mesh.triangles)[tri]]
    return jnp.sum(dv * bary, axis=-1), tri, inside


def trace(mesh: M.Mesh, d, pose, side: int, u, iters: int = 3):
    r"""Seabed curve of one side: locations :math:`q_k` (K, 2) where the ping plane meets the seabed.

    The ping plane passes through the sonar with normal :math:`h = R e_x`. Samples step
    :math:`u_k` horizontally outboard and are shifted :math:`w_k` along track to stay in the
    plane, :math:`w = -h_z (z_\mathrm{ref} - d(q) - s_z) / \lVert h_{xy} \rVert`, iterated
    because :math:`d(q)` depends on :math:`w`.
    """
    s, R = pose_frame(pose)
    hd = R[:, 0]
    nxy = jnp.linalg.norm(hd[:2])
    fwd = hd[:2] / nxy
    out = jnp.array([-fwd[1], fwd[0]]) if side == PORT else jnp.array([fwd[1], -fwd[0]])
    u = jnp.asarray(u)
    w = jnp.zeros_like(u)
    for _ in range(iters):
        q = s[:2] + u[:, None] * out + w[:, None] * fwd
        z = mesh.z_ref - _depth(mesh, d, q)[0]
        w = -hd[2] * (z - s[2]) / nxy
    return s[:2] + u[:, None] * out + w[:, None] * fwd


def samples(mesh: M.Mesh, d, pose, side: int, u):
    r"""Seabed points :math:`P_k` (K, 3), upward face normals :math:`n_k` (K, 3), and an on-mesh mask (K,)."""
    q = trace(mesh, d, pose, side, u)
    depth, tri, inside = _depth(mesh, d, q)
    g = M.face_gradient(mesh, d)[tri]
    n = jnp.concatenate([g, jnp.ones_like(g[:, :1])], axis=-1)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    P = jnp.concatenate([q, (mesh.z_ref - depth)[:, None]], axis=-1)
    return P, n, inside


def echo(theta: Theta, pose, side: int, P, n):
    r"""Echo strength :math:`a_c` (K,) and range :math:`r` (K,) of seabed points P with normals n.

    .. math::

        a_c = B(\beta_c) \cos^2\varphi \, \frac{e^{-4 \alpha r}}{r^3}, \qquad
        \cos\varphi = \max(0, -n^\top e),

    without the visibility, which the caller applies. Faces seen from below get zero.
    """
    s, R = pose_frame(pose)
    axes = R @ jnp.asarray(R_MOUNT[side])
    v = P - s
    r = jnp.linalg.norm(v, axis=-1)
    e = v / r[:, None]
    beta = jnp.arctan2(e @ axes[:, 2], e @ axes[:, 1])
    cos_phi = jnp.maximum(-jnp.sum(n * e, axis=-1), 0.0)
    B = jnp.exp(-4 * math.log(2) * (beta - theta.beta0) ** 2 / theta.omega**2)
    return B * cos_phi**2 * jnp.exp(-4 * theta.alpha * r) / r**3, r


def visibility(mesh: M.Mesh, d, pose, side: int, u, step: float = 0.25, margin: float = 0.5) -> np.ndarray:
    r"""Visibility :math:`V` (K,) bool: whether the straight ray to each curve sample clears the seabed.

    The ray is sampled every `step` m horizontally and is shadowed where the seabed rises
    above it, except within `margin` m of the sample, so a face does not shadow itself.
    Seabed off the mesh never shadows. Not differentiable; the outer loop holds it fixed.
    """
    P, _, _ = samples(mesh, d, pose, side, u)
    s = np.asarray(pose_frame(pose)[0])
    P = np.asarray(P)
    horiz = np.hypot(P[:, 0] - s[0], P[:, 1] - s[1])
    if len(P) == 0:
        return np.zeros(0, bool)
    t = np.linspace(0.0, 1.0, max(math.ceil(horiz.max() / step), 1) + 1)[1:-1]
    xy = s[None, None, :2] + t[None, :, None] * (P[:, None, :2] - s[None, None, :2])
    ray_z = s[2] + t[None, :] * (P[:, None, 2] - s[2])
    before = t[None, :] * horiz[:, None] < horiz[:, None] - margin
    seabed_z = mesh.z_ref - np.asarray(M.interpolate(mesh, d, xy))  # NaN off the mesh compares False
    return ~((seabed_z > ray_z) & before).any(axis=1)
