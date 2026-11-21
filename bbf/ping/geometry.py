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
import math

import jax
import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.ping.theta import PORT, Theta

# Proper rotations in the sonar frame whose :math:`e_y` points outboard and :math:`e_z` down.
R_MOUNT = np.stack([np.diag([-1.0, 1, -1]),   # port: :math:`180^\circ` about the port axis
                    np.diag([1.0, -1, -1])])  # starboard: :math:`180^\circ` about the forward axis
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
    """Depth at q and whether q is on the mesh; finite everywhere, but meaningless off the mesh."""
    tri, bary, inside = M.locate(mesh, q)
    dv = jnp.asarray(d)[jnp.asarray(mesh.triangles)[tri]]
    return jnp.sum(dv * bary, axis=-1), tri, inside


def field_at(mesh: M.Mesh, x, q):
    r"""Hat-function interpolation :math:`\sum_i x_i \psi_i(q)` of vertex values x at q (K, 2); finite off the mesh."""
    return _depth(mesh, x, q)[0]


def side_frame(pose, side: int):
    r"""Sonar position :math:`s`, heading :math:`h = R e_x`, :math:`\lVert h_{xy} \rVert`, and the horizontal forward and outboard unit axes of a side."""
    s, R = pose_frame(pose)
    hd = R[:, 0]
    nxy = jnp.linalg.norm(hd[:2])
    fwd = hd[:2] / nxy
    out = jnp.array([-fwd[1], fwd[0]]) if side == PORT else jnp.array([fwd[1], -fwd[0]])
    return s, hd, nxy, fwd, out


def trace(mesh: M.Mesh, d, pose, side: int, u, iters: int = 3):
    r"""Seabed curve of one side: locations :math:`q_k` (K, 2) where the ping plane meets the seabed.

    The ping plane passes through the sonar with normal :math:`h = R e_x`. Samples step
    :math:`u_k` horizontally outboard and are shifted :math:`w_k` along track to stay in the
    plane, :math:`w = -h_z (z_\mathrm{ref} - d(q) - s_z) / \lVert h_{xy} \rVert`, iterated
    because :math:`d(q)` depends on :math:`w`.
    """
    s, hd, nxy, fwd, out = side_frame(pose, side)
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
    B = jnp.exp(-4 * math.log(2) * (beta - theta.beta0[side]) ** 2 / theta.omega[side] ** 2)
    return B * cos_phi**2 * jnp.exp(-4 * theta.alpha * r) / r**3, r


def knife_edge(nu):
    r"""Intensity factor of knife-edge diffraction at Fresnel parameter :math:`\nu` (ITU-R P.526).

    .. math::

        J(\nu) = 6.9 + 20 \log_{10}\Bigl(\sqrt{(\nu - 0.1)^2 + 1} + \nu - 0.1\Bigr)\ \mathrm{dB}
        \quad (\nu > -0.78), \qquad J = 0 \text{ otherwise},

    returned as :math:`10^{-J/10}`: 1 in the open, :math:`\approx 1/4` at grazing incidence
    (:math:`\nu = 0`), and falling as :math:`\approx 0.05 / \nu^2` deep in shadow.
    """
    J = 6.9 + 20 * jnp.log10(jnp.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    return jnp.where(nu > -0.78, 10 ** (-J / 10), 1.0)


def _running_max(alpha):
    """Running maximum of alpha (K,) and the index where it was reached."""
    def pick(a, b):
        take_b = b[0] >= a[0]
        return jnp.where(take_b, b[0], a[0]), jnp.where(take_b, b[1], a[1])
    return jax.lax.associative_scan(pick, (alpha, jnp.arange(alpha.shape[0])))


def elevation(P, s, inside=None):
    r"""Elevation angle :math:`\alpha_k = \operatorname{atan2}(z_k - s_z, \lVert q_k - s_{xy} \rVert)` (K,) of points P (K, 3) seen from s.

    Points off the mesh (`inside` False) get :math:`-\pi/2`, so they never cast a horizon: their
    depth is not the seabed's.
    """
    alpha = jnp.arctan2(P[:, 2] - s[2], jnp.linalg.norm(P[:, :2] - s[:2], axis=-1))
    return alpha if inside is None else jnp.where(inside, alpha, -jnp.pi / 2)


def horizon_visibility(P, s, u, margin: float, wavelength: float, inside=None):
    r"""Visibility :math:`V_k` (K,) of curve samples P (K, 3) from the sonar at s, with diffraction.

    The ray to a sample lies in the ping plane, as does the curve. Seen from the sonar,
    sample :math:`k` has elevation angle
    :math:`\alpha_k = \operatorname{atan2}(z_k - s_z, \lVert q_k - s_{xy} \rVert)`, and the samples
    at least `margin` closer cast the horizon :math:`H_k = \alpha_{j^*} = \max_j \alpha_j`. The
    crest :math:`j^*` sits at distances :math:`d_1 = \lVert P_{j^*} - s \rVert` from the sonar and
    :math:`d_2 = \lVert P_k - P_{j^*} \rVert` from the sample, and rises
    :math:`h = d_1 \sin(H_k - \alpha_k)` above the line of sight (negative when the sample is
    in the open). With wavelength :math:`\lambda`,

    .. math::

        \nu_k = h \sqrt{\frac{2 (d_1 + d_2)}{\lambda d_1 d_2}}, \qquad V_k = \mathrm{knife\_edge}(\nu_k).

    Diffraction makes the shadow edge soft on the scale of the Fresnel zone and leaves a
    small echo deep in shadow whose relative change with depth stays finite, so gradients
    reach the crest and the seabed it hides. The margin keeps a sample from being shadowed
    by its own face. Samples must be ordered outboard by u (K,); those off the mesh (`inside`
    False) cast no horizon.
    """
    alpha = elevation(P, s, inside)
    u = np.asarray(u)
    idx = np.searchsorted(u, u - margin, side="right") - 1  # last sample at least `margin` closer
    has = jnp.asarray(idx >= 0)
    peak, at = _running_max(alpha)
    j = at[np.maximum(idx, 0)]
    d1 = jnp.linalg.norm(P[j] - s, axis=-1)
    # samples without a horizon may coincide with P[j]; keep the norm's gradient finite there
    gap = jnp.where(has[:, None], P - P[j], 1.0)
    d2 = jnp.maximum(jnp.linalg.norm(gap, axis=-1), 1e-6)
    h = d1 * jnp.sin(peak[np.maximum(idx, 0)] - alpha)
    nu = h * jnp.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))
    return jnp.where(has, knife_edge(nu), 1.0)
