r"""First-return factor: each ping's first seabed echo fixes the range of the nearest seabed.

The data's first echo of ping t is :math:`r_{0,t} = \min_c r_{0,t,c}` over its two sides
(`bbf.inference.init.first_echo`): the slant range of the seabed point nearest to the sonar in
the ping's plane, on either side. The ping factor cannot pin it: under the track the beam
(centred tens of degrees off nadir) predicts little energy, so the depths there barely change
the rendered intensities and are otherwise free to run away.

One factor per ping, not per side: both sides' curves start at the nadir, so neither side's
nearest seabed can be farther than the other's nadir; a side whose detector fires late (on
the real data, port's on pass A by 0.3 m on median, with a tail of metres) would otherwise
pull its half of the seabed down. The earlier side's echo is the one to trust.

The nearest range is a soft minimum over both sides' curve samples (`geometry.trace`),
:math:`\tilde r = \sum_k w_k r_k` with :math:`w = \operatorname{softmax}(-r / \tau)` over the on-mesh
samples, which is within about :math:`\tau / 2` of :math:`\min_k r_k` (up to :math:`\tau` where the
minimum is at nadir, the curves' end). The factor is a Gaussian scale mixture,
:math:`p(r_0 \mid d) = (1 - \epsilon)\, \mathcal{N}(r_0; \tilde r, \sigma^2) + \epsilon\, \mathcal{N}(r_0; \tilde r, \sigma_w^2)`:
the wide component takes detector errors without letting a large violation go free, as a
uniform floor would (its gradient vanishes, so a runaway vertex would not be pulled back).

The derivative :math:`\partial \tilde r / \partial d` is taken with the curve's horizontal samples
held fixed (exact for a level ping plane; to second order otherwise, since :math:`r` is
stationary along the curve at its minimum):
:math:`\partial \tilde r = \sum_k w_k \bigl(1 - (r_k - \tilde r) / \tau\bigr)\, \partial r_k` and
:math:`\partial r_k / \partial d(q_k) = (s_z - P_{k,z}) / r_k`, spread over the sample's triangle
by its barycentric weights. Each ping thus touches a few vertices near its nearest point.
"""
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from bbf import mesh as M
from bbf.ping import geometry as G

SIGMA, SIGMA_WIDE, EPS, TAU, DU = 0.15, 2.0, 0.2, 0.02, 0.05


@dataclass(frozen=True, eq=False)
class FirstReturn:
    r"""First echoes r0 per pass, (n, 2) per side or (n,) per ping (NaN where none was found), and the factor's constants.

    sigma, sigma_wide, eps: standard deviations (m) of the detected first echo about the nearest
    seabed range, and the weight of the wide one; tau: soft-min temperature (m); du: horizontal
    step (m) of the curve samples, out to the largest :math:`r_0` (the nearest point is never
    farther out horizontally than its slant range).
    """
    r0: list[np.ndarray]
    sigma: float = SIGMA
    sigma_wide: float = SIGMA_WIDE
    eps: float = EPS
    tau: float = TAU
    du: float = DU

    @cached_property
    def per_ping(self) -> list[np.ndarray]:
        r""":math:`r_{0,t}` per pass (n,): the earlier side's first echo, NaN where neither side has one."""
        out = []
        for r in self.r0:
            r = np.asarray(r, dtype=float)
            if r.ndim == 2:
                fin = np.isfinite(r)
                r = np.where(fin.any(axis=1), np.min(np.where(fin, r, np.inf), axis=1), np.nan)
            out.append(r)
        return out

    @cached_property
    def u(self) -> np.ndarray:
        r_max = max(np.nanmax(r, initial=1.0) for r in self.per_ping)
        return np.arange(self.du / 2, r_max + 1.0, self.du)


def nearest(mesh: M.Mesh, d, pose, u, tau: float):
    r"""Soft-min range :math:`\tilde r` over both sides, whether any sample is on the mesh, and the sparse :math:`\partial \tilde r / \partial d`.

    Returns (r, ok, vid (2K, 3), coef (2K, 3)): :math:`\partial \tilde r / \partial d_v` is the sum
    of coef over the entries where vid is v.
    """
    q = jnp.concatenate([G.trace(mesh, d, pose, side, u) for side in (0, 1)])
    tri, bary, inside = M.locate(mesh, q)
    vid = jnp.asarray(mesh.triangles)[tri]
    depth = jnp.sum(jnp.asarray(d)[vid] * bary, axis=-1)
    s = jnp.asarray(pose)[:3]
    Pz = mesh.z_ref - depth
    r = jnp.sqrt(jnp.sum((q - s[:2]) ** 2, axis=-1) + (Pz - s[2]) ** 2)
    ok = inside.any()
    logits = jnp.where(inside, -r / tau, -jnp.inf)
    w = jax.nn.softmax(jnp.where(ok, logits, 0.0))
    w = jnp.where(inside, w, 0.0)
    rt = jnp.sum(w * r)
    dr_k = (s[2] - Pz) / r  # :math:`\partial r_k / \partial d(q_k)`
    c = w * (1.0 - (r - rt) / tau) * dr_k
    return rt, ok, vid, c[:, None] * bary


def log_mix(r0, rt, fr: FirstReturn):
    r""":math:`\log p(r_0 \mid \tilde r)`, its derivative in :math:`\tilde r`, and the Fisher weight :math:`\sum_i \gamma_i / \sigma_i^2` (responsibilities :math:`\gamma_i`)."""
    e = np.asarray(r0) - np.asarray(rt)
    comps = [(1 - fr.eps, fr.sigma), (fr.eps, fr.sigma_wide)]
    lc = np.stack([math.log(a) - 0.5 * (e / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi) for a, s in comps])
    lp = np.logaddexp(lc[0], lc[1])
    resp = np.exp(lc - lp)
    prec = np.array([1 / s**2 for _, s in comps])[:, None]
    return lp, (resp * prec).sum(0) * e, (resp * prec).sum(0)


@lru_cache(maxsize=8)
def _batched(mesh: M.Mesh, tau: float):
    return jax.jit(jax.vmap(lambda d, pose, u: nearest(mesh, d, pose, u, tau), in_axes=(None, 0, None)))


def first_return_term(mesh: M.Mesh, d, poses: list[np.ndarray], fr: FirstReturn, chunk: int = 256):
    r"""Over all passes: :math:`\sum_t \log p(r_{0,t} \mid d)`, its gradient in d (n,), and the GGN rows.

    Returns (value, grad, A, W): A (m, n) sparse, one row :math:`\partial \tilde r / \partial d` per
    scored ping, and W (m,) their Fisher weights, so the term's GGN is :math:`A^\top W A`.
    """
    n = mesh.n_vertices
    d = jnp.asarray(d, dtype=float)
    u = jnp.asarray(fr.u)
    f = _batched(mesh, fr.tau)
    total, grad = 0.0, np.zeros(n)
    rows, cols, vals, weights = [], [], [], []
    m = 0
    for P, r0 in zip(poses, fr.per_ping, strict=True):
        for i in range(0, len(P), chunk):
            rt, ok, vid, coef = (np.asarray(a) for a in f(d, jnp.asarray(P[i:i + chunk]), u))
            obs = r0[i:i + chunk]
            use = ok & np.isfinite(obs)
            if not use.any():
                continue
            lp, g, W = log_mix(obs[use], rt[use], fr)
            total += float(lp.sum())
            vid, coef = vid[use].reshape(len(lp), -1), coef[use].reshape(len(lp), -1)
            np.add.at(grad, vid, g[:, None] * coef)
            keep = coef != 0
            rr = np.broadcast_to(np.arange(m, m + len(lp))[:, None], coef.shape)
            rows.append(rr[keep])
            cols.append(vid[keep])
            vals.append(coef[keep])
            weights.append(W)
            m += len(lp)
    if m == 0:
        return total, grad, sp.csr_matrix((0, n)), np.zeros(0)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, n))
    return total, grad, A, np.concatenate(weights)
