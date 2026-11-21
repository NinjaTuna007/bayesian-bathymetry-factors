r"""First-return factor: each ping-side's first seabed echo fixes the range of its nearest seabed.

The data's first echo :math:`r_{0,t,c}` (`bbf.inference.init.first_echo`, which removes the other
channel's leak) is the slant range of the seabed point nearest to the sonar in ping t's plane on
side c. The ping factor cannot pin it: under the track the beam (centred tens of degrees off
nadir) predicts little energy, so the depths there barely change the rendered intensities and
are otherwise free to run away.

Per side: both sides' curves start at the nadir, so each side's nearest range is at most the
range below the sonar, but on a slope the up-slope side's comes from the bank while the other
side's is the floor (pass A: starboard at 16.7 m on its bank, port at about 21 m on the floor).
Both bound the seabed, so both are kept.

The nearest range is a soft minimum over the side's curve samples (`geometry.trace`),
:math:`\tilde r = \sum_k w_k r_k` with :math:`w = \operatorname{softmax}(-r / \tau)` over the on-mesh
samples, which is within about :math:`\tau / 2` of :math:`\min_k r_k` (up to :math:`\tau` where the
minimum is at nadir, the curve's end). The factor is a mixture,
:math:`p(r_0 \mid d) = (1 - \epsilon - \epsilon_o)\, \mathcal{N}(r_0; \tilde r, \sigma^2) + \epsilon\, \mathcal{N}(r_0; \tilde r, \sigma_w^2) + \epsilon_o / R`:
the wide component takes detector errors of a metre or two and still pulls back a seabed that
strays (a uniform floor alone would let a runaway vertex go free), and the small uniform one
takes echoes picked on the wrong feature altogether (on A, port's floor is ambiguous on some
pings).

The derivative :math:`\partial \tilde r / \partial d` is taken with the curve's horizontal samples
held fixed (exact for a level ping plane; to second order otherwise, since :math:`r` is
stationary along the curve at its minimum):
:math:`\partial \tilde r = \sum_k w_k \bigl(1 - (r_k - \tilde r) / \tau\bigr)\, \partial r_k` and
:math:`\partial r_k / \partial d(q_k) = (s_z - P_{k,z}) / r_k`, spread over the sample's triangle
by its barycentric weights.
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

SIGMA, SIGMA_WIDE, EPS, EPS_OUT, TAU, DU, ROW_TOL = 0.15, 2.0, 0.2, 0.02, 0.02, 0.05, 1e-2


@dataclass(frozen=True, eq=False)
class FirstReturn:
    r"""First echoes r0 per pass (n, 2) (NaN where none was found) and the factor's constants.

    sigma, sigma_wide, eps: standard deviations (m) of the detected first echo about the nearest
    seabed range, and the weight of the wide one; eps_out: weight of the uniform outlier
    component over the largest :math:`r_0`; tau: soft-min temperature (m); du: horizontal
    step (m) of the curve samples, out to the largest :math:`r_0` (the nearest point is never
    farther out horizontally than its slant range); row_tol: GGN row entries below this fraction
    of the row's largest are dropped (the gradient keeps them all).
    """
    r0: list[np.ndarray]
    sigma: float = SIGMA
    sigma_wide: float = SIGMA_WIDE
    eps: float = EPS
    eps_out: float = EPS_OUT
    tau: float = TAU
    du: float = DU
    row_tol: float = ROW_TOL

    @cached_property
    def r_max(self) -> float:
        return float(max(np.nanmax(r, initial=1.0) for r in self.r0))

    @cached_property
    def u(self) -> np.ndarray:
        return np.arange(self.du / 2, self.r_max + 1.0, self.du)


def nearest(mesh: M.Mesh, d, pose, side: int, u, tau: float):
    r"""Soft-min range :math:`\tilde r` of one side, whether any sample is on the mesh, and the sparse :math:`\partial \tilde r / \partial d`.

    Returns (r, ok, vid (K, 3), coef (K, 3)): :math:`\partial \tilde r / \partial d_v` is the sum
    of coef over the entries where vid is v.
    """
    q = G.trace(mesh, d, pose, side, u)
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
    comps = [(1 - fr.eps - fr.eps_out, fr.sigma), (fr.eps, fr.sigma_wide)]
    lc = np.stack([math.log(a) - 0.5 * (e / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi) for a, s in comps]
                  + [np.full(e.shape, math.log(fr.eps_out / fr.r_max))])
    lp = np.logaddexp.reduce(lc, axis=0)
    resp = np.exp(lc[:2] - lp)
    prec = np.array([1 / s**2 for _, s in comps])[:, None]
    return lp, (resp * prec).sum(0) * e, (resp * prec).sum(0)


@lru_cache(maxsize=8)
def _batched(mesh: M.Mesh, side: int, tau: float):
    return jax.jit(jax.vmap(lambda d, pose, u: nearest(mesh, d, pose, side, u, tau), in_axes=(None, 0, None)))


def first_return_term(mesh: M.Mesh, d, poses: list[np.ndarray], fr: FirstReturn, chunk: int = 256):
    r"""Over all passes: :math:`\sum_{t,c} \log p(r_{0,t,c} \mid d)`, its gradient in d (n,), and the GGN rows.

    Returns (value, grad, A, W): A (m, n) sparse, one row :math:`\partial \tilde r / \partial d` per
    scored ping-side, and W (m,) their Fisher weights, so the term's GGN is :math:`A^\top W A`.
    """
    n = mesh.n_vertices
    d = jnp.asarray(d, dtype=float)
    u = jnp.asarray(fr.u)
    total, grad = 0.0, np.zeros(n)
    rows, cols, vals, weights = [], [], [], []
    m = 0
    for P, r0 in zip(poses, fr.r0, strict=True):
        for side in (0, 1):
            f = _batched(mesh, side, fr.tau)
            for i in range(0, len(P), chunk):
                rt, ok, vid, coef = (np.asarray(a) for a in f(d, jnp.asarray(P[i:i + chunk]), u))
                obs = np.asarray(r0, dtype=float)[i:i + chunk, side]
                use = ok & np.isfinite(obs)
                if not use.any():
                    continue
                lp, g, W = log_mix(obs[use], rt[use], fr)
                total += float(lp.sum())
                vid, coef = vid[use].reshape(len(lp), -1), coef[use].reshape(len(lp), -1)
                np.add.at(grad, vid, g[:, None] * coef)
                # the GGN keeps each row's entries above `fr.row_tol` of its largest: near a flat nadir
                # the soft-min spreads over metres of curve, and :math:`A^\top W A` would fill a dense block
                # of hundreds of vertices per ping
                keep = np.abs(coef) > fr.row_tol * np.abs(coef).max(axis=1, keepdims=True)
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
