r"""The log posterior of the depths and its gradient, through the rendered unit intensities.

Every data factor depends on the depths only through the unit intensities :math:`S` of the
pings (`bbf.ping.unit_intensity`): the ping factor through each ping-side's :math:`S`, the
shadow factor through block means of :math:`S`, and the match factor through :math:`S` at the
pair bins. So

.. math::

    \ell(d, \beta) = \log p(\beta) + \log p(d \mid \beta) + F\bigl(S(d)\bigr),

and :math:`\nabla_d F = J^\top \nabla_S F` with :math:`J = \partial S / \partial d`. :math:`\nabla_S F`
is one JAX gradient over all pings; :math:`J^\top` is applied in chunks of pings, so memory
stays that of one chunk.
"""
from dataclasses import dataclass, field
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.match.factor import log_match_factor
from bbf.match.pairs import Pairs, matched_mask
from bbf.ping.factor import _objective, mode
from bbf.ping.render import unit_intensity
from bbf.ping.shadow import log_shadow_side, ping_use
from bbf.ping.theta import Theta
from bbf.prior import Prior


@dataclass(eq=False)  # hashed by identity, so compiled functions can be cached per problem
class Problem:
    """Everything the objective needs, with the outer loop's choices (labels, pairs) fixed.

    poses: per pass (n, 6), mesh-local; y: per pass (n, 2, B) intensities; labels: per pass
    (n, 2, n_blocks) lit/shadow labels; pairs: matched pairs keyed by pass indices (a, b).
    """
    mesh: M.Mesh
    prior: Prior
    theta: Theta
    poses: list[np.ndarray]
    y: list[np.ndarray]
    labels: list[np.ndarray]
    u: np.ndarray
    dr: float
    pairs: dict[tuple[int, int], Pairs] = field(default_factory=dict)
    chunk: int = 32

    @property
    def n_bins(self) -> int:
        return self.y[0].shape[-1]

    def uses(self) -> list[np.ndarray]:
        """Per pass (n, 2, B): bins the ping factor scores (lit blocks, minus matched bins)."""
        out = []
        for p, (y, lab) in enumerate(zip(self.y, self.labels)):
            matched = np.zeros(y.shape, bool)
            for (a, b), pr in self.pairs.items():
                if a == p:
                    matched |= matched_mask(pr, len(y), self.n_bins, "a")
                if b == p:
                    matched |= matched_mask(pr, len(y), self.n_bins, "b")
            out.append(np.stack([ping_use(lab[t], self.n_bins, self.theta.block, matched[t]) for t in range(len(y))]))
        return out


def _chunks(n: int, size: int):
    """Index arrays of fixed length `size` covering 0..n-1; the last is padded with its final index."""
    for s in range(0, n, size):
        idx = np.arange(s, min(s + size, n))
        yield idx, np.concatenate([idx, np.full(size - len(idx), idx[-1])])


def _render_chunk(pb: Problem, d, poses):
    return jnp.stack([jax.vmap(lambda pose, side=side: unit_intensity(pb.mesh, d, pose, side, pb.theta, pb.u,
                                                                      pb.n_bins, pb.dr))(poses)
                      for side in (0, 1)], axis=1)


@cache
def _render_jit(pb: Problem):
    return jax.jit(lambda d, poses: _render_chunk(pb, d, poses))


@cache
def _vjp_jit(pb: Problem):
    def chunk_vjp(d, poses, g):
        _, vjp = jax.vjp(lambda d: _render_chunk(pb, d, poses), d)
        return vjp(g)[0]
    return jax.jit(chunk_vjp)


def render(problem: Problem, d) -> list[np.ndarray]:
    """Unit intensities (n, 2, B) of every ping of every pass at depths d."""
    pb = problem
    f = _render_jit(pb)
    d = jnp.asarray(d)
    out = []
    for poses in pb.poses:
        S = np.zeros((len(poses), 2, pb.n_bins))
        for idx, padded in _chunks(len(poses), pb.chunk):
            S[idx] = np.asarray(f(d, jnp.asarray(poses[padded])))[: len(idx)]
        out.append(S)
    return out


def data_term(problem: Problem):
    r"""Jitted :math:`F(S)` over all passes: (value, (:math:`\hat u` per pass (n, 2), :math:`\hat\mu` per pair))."""
    pb = problem
    th, dr = pb.theta, pb.dr
    uses = [jnp.asarray(u) for u in pb.uses()]
    ys = [jnp.asarray(y) for y in pb.y]
    labels = [jnp.asarray(lab) for lab in pb.labels]
    keys = sorted(pb.pairs)

    def side_terms(side, y, S, use, lab):
        Fu, _, _ = _objective(th, side, dr, y, S, use)
        u_hat, H = mode(th, side, dr, y, S, use)
        ping = Fu(u_hat) + 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(H)
        return ping + log_shadow_side(th, side, lab, S, u_hat), u_hat

    def F(S_list):
        total, u_hats = 0.0, []
        for y, S, use, lab in zip(ys, S_list, uses, labels):
            us = []
            for side in (0, 1):
                v, uh = jax.vmap(lambda yy, SS, uu, ll, side=side: side_terms(side, yy, SS, uu, ll))(
                    y[:, side], S[:, side], use[:, side], lab[:, side])
                total = total + v.sum()
                us.append(uh)
            u_hats.append(jnp.stack(us, axis=1))
        mus = []
        for a, b in keys:
            v, mu = log_match_factor(pb.pairs[(a, b)], S_list[a], S_list[b], ys[a], ys[b], th, dr)
            total = total + v
            mus.append(mu)
        return total, (u_hats, mus)

    return jax.jit(F)


def data_vjp(problem: Problem, d, g_S: list[np.ndarray]) -> np.ndarray:
    r""":math:`J^\top g_S`: the gradient with respect to d of :math:`\sum g_S \cdot S(d)`, chunked over pings."""
    pb = problem
    f = _vjp_jit(pb)
    d = jnp.asarray(d)
    grad = np.zeros(pb.mesh.n_vertices)
    for poses, g in zip(pb.poses, g_S):
        for idx, padded in _chunks(len(poses), pb.chunk):
            gc = np.zeros((pb.chunk, 2, pb.n_bins))
            gc[: len(idx)] = g[idx]
            grad += np.asarray(f(d, jnp.asarray(poses[padded]), jnp.asarray(gc)))
    return grad


def value_and_grad(problem: Problem, d, beta, F=None):
    r""":math:`\ell(d, \beta)`, its gradient (d, :math:`\beta`), and the data term's details.

    F: a `data_term(problem)` to reuse (it is jitted once per problem); details hold the
    rendered S, :math:`\hat u` per pass and :math:`\hat\mu` per pair.
    """
    F = data_term(problem) if F is None else F
    S = render(problem, d)
    (val, (u_hats, mus)), g_S = jax.value_and_grad(F, has_aux=True)([jnp.asarray(s) for s in S])
    grad_d = data_vjp(problem, d, [np.asarray(g) for g in g_S])
    gd_prior, gb_prior = problem.prior.grad(np.asarray(d), np.asarray(beta))
    value = float(val) + problem.prior.log_density(np.asarray(d), np.asarray(beta))
    details = {"S": S, "u": [np.asarray(u) for u in u_hats], "mu": [float(m) for m in mus], "data": float(val)}
    return value, (grad_d + gd_prior, gb_prior), details
