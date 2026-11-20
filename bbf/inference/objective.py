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

The sonar parameters :math:`\psi` (beam centre and width, linear gain; `bbf.ping.theta.psi_of`)
are an argument too: the beam reaches :math:`F` through :math:`S`, the gain directly. With
`Problem.psi_prior` set they are unknowns of the graph with a Gaussian prior, and the gradient
includes them; otherwise they stay at the values in `Problem.theta`.
"""
from dataclasses import dataclass, field
from functools import cache, cached_property, lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.match.factor import log_match_factor
from bbf.match.pairs import Pairs, matched_mask
from bbf.ping.factor import WARM_ITERS, laplace_ping_side
from bbf.ping.render import unit_intensity
from bbf.ping.shadow import log_shadow_side, ping_use
from bbf.ping.theta import Theta, psi_of, render_key, render_theta, with_psi
from bbf.prior import Prior


@dataclass(frozen=True, eq=False)  # hashed by identity, so the data term can be cached per problem
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
    valid: list[np.ndarray] | None = None  # per pass (n, 2, n_blocks): blocks to use (see onset_valid)
    psi_prior: tuple[np.ndarray, np.ndarray] | None = None  # (mean, sd) of psi (2, 3); None: psi fixed

    @property
    def free_psi(self) -> bool:
        return self.psi_prior is not None

    def psi_log_prior(self, psi) -> tuple[float, np.ndarray]:
        r"""Gaussian log prior of :math:`\psi` and its gradient (2, 3); zero when :math:`\psi` is fixed."""
        if self.psi_prior is None:
            return 0.0, np.zeros((2, 3))
        mean, sd = (np.asarray(a, float) for a in self.psi_prior)
        z = (np.asarray(psi) - mean) / sd
        return float(-0.5 * np.sum(z**2) - np.sum(np.log(sd)) - z.size * 0.5 * np.log(2 * np.pi)), -z / sd

    @property
    def n_bins(self) -> int:
        return self.y[0].shape[-1]

    def uses(self) -> list[np.ndarray]:
        """Per pass (n, 2, B): bins the ping factor scores (lit blocks, minus matched bins)."""
        return self._uses

    @cached_property
    def _uses(self) -> list[np.ndarray]:
        out = []
        for p, (y, lab) in enumerate(zip(self.y, self.labels, strict=True)):
            matched = np.zeros(y.shape, bool)
            for (a, b), pr in self.pairs.items():
                if a == p:
                    matched |= matched_mask(pr, len(y), self.n_bins, "a")
                if b == p:
                    matched |= matched_mask(pr, len(y), self.n_bins, "b")
            val = None if self.valid is None else self.valid[p]
            out.append(np.stack([ping_use(lab[t], self.n_bins, self.theta.block, matched[t],
                                          None if val is None else val[t]) for t in range(len(y))]))
        return out


def _chunks(n: int, size: int):
    """Index arrays of fixed length `size` covering 0..n-1; the last is padded with its final index."""
    for s in range(0, n, size):
        idx = np.arange(s, min(s + size, n))
        yield idx, np.concatenate([idx, np.full(size - len(idx), idx[-1])])


def _key(pb: Problem) -> tuple:
    """What the compiled render depends on: the mesh, theta's render fields, the samples and bins."""
    return (pb.mesh, render_key(pb.theta), np.asarray(pb.u).tobytes(), pb.n_bins, pb.dr)


def _render_chunk(key: tuple, d, poses, psi):
    mesh, rkey, u_bytes, n_bins, dr = key
    th = render_theta(rkey, psi)
    u = np.frombuffer(u_bytes)
    return jnp.stack([jax.vmap(lambda pose, side=side: unit_intensity(mesh, d, pose, side, th, u, n_bins, dr))(poses)
                      for side in (0, 1)], axis=1)


@cache
def _render_jit(key: tuple):
    return jax.jit(lambda d, poses, psi: _render_chunk(key, d, poses, psi))


@cache
def _vjp_jit(key: tuple):
    def chunk_vjp(d, poses, psi, g):
        _, vjp = jax.vjp(lambda d, psi: _render_chunk(key, d, poses, psi), d, psi)
        return vjp(g)
    return jax.jit(chunk_vjp)


@cache
def _beam_jvp_jit(key: tuple):
    def chunk_jvp(d, poses, psi):
        # side c's S depends only on side c's beam, so one tangent per column covers both sides
        cols = []
        for k in (0, 1):
            t = jnp.zeros_like(psi).at[:, k].set(1.0)
            cols.append(jax.jvp(lambda p: _render_chunk(key, d, poses, p), (psi,), (t,))[1])
        return jnp.stack(cols, axis=-1)
    return jax.jit(chunk_jvp)


def _psi(problem: Problem, psi):
    return jnp.asarray(psi_of(problem.theta) if psi is None else psi, dtype=float)


def render(problem: Problem, d, psi=None) -> list[np.ndarray]:
    r"""Unit intensities (n, 2, B) of every ping of every pass at depths d (and :math:`\psi`, default theta's)."""
    pb = problem
    f = _render_jit(_key(pb))
    d, psi = jnp.asarray(d), _psi(pb, psi)
    out = []
    for poses in pb.poses:
        S = np.zeros((len(poses), 2, pb.n_bins))
        for idx, padded in _chunks(len(poses), pb.chunk):
            S[idx] = np.asarray(f(d, jnp.asarray(poses[padded]), psi))[: len(idx)]
        out.append(S)
    return out


def beam_jacobian(problem: Problem, d, psi=None) -> list[np.ndarray]:
    r"""Per pass (n, 2, B, 2): :math:`\partial S / \partial(\beta_0, \omega)` of each ping-side, with respect to its own side's beam."""
    pb = problem
    f = _beam_jvp_jit(_key(pb))
    d, psi = jnp.asarray(d), _psi(pb, psi)
    out = []
    for poses in pb.poses:
        Jb = np.zeros((len(poses), 2, pb.n_bins, 2))
        for idx, padded in _chunks(len(poses), pb.chunk):
            Jb[idx] = np.asarray(f(d, jnp.asarray(poses[padded]), psi))[: len(idx)]
        out.append(Jb)
    return out


@lru_cache(maxsize=8)  # compiled once per problem (hashed by identity); a round uses a few
def data_term(problem: Problem, warm: bool = False):
    r"""Jitted :math:`F(S, \hat u_0, \psi)` over all passes: (value, (:math:`\hat u` per pass (n, 2), :math:`\hat\mu` per pair)).

    :math:`\hat u_0` and :math:`\psi` may be None (cold modes; theta's :math:`\psi`). With `warm`,
    each ping-side's log-kappa mode starts from
    :math:`\hat u_0` (per pass (n, 2), e.g. the previous evaluation's) and takes only a few
    Newton steps, which is enough within a Gauss--Newton loop where the modes move little.
    """
    pb = problem
    dr = pb.dr
    uses = [jnp.asarray(u) for u in pb.uses()]
    ys = [jnp.asarray(y) for y in pb.y]
    labels = [jnp.asarray(lab) for lab in pb.labels]
    valids = [jnp.ones(lab.shape, bool) for lab in labels] if pb.valid is None else [jnp.asarray(v) for v in pb.valid]
    keys = sorted(pb.pairs)

    def side_terms(th, side, y, S, use, lab, val, u0):
        if warm:
            ping, u_hat = laplace_ping_side(th, side, dr, y, S, use, u0=u0, iters=WARM_ITERS)
        else:
            ping, u_hat = laplace_ping_side(th, side, dr, y, S, use)
        return ping + log_shadow_side(th, side, lab, S, u_hat, val), u_hat

    psi0 = _psi(pb, None)

    def F(S_list, u0_list=None, psi=None):
        th = with_psi(pb.theta, psi0 if psi is None else psi)
        total, u_hats = 0.0, []
        if u0_list is None:
            u0_list = [jnp.zeros(y.shape[:2]) for y in ys]
        for y, S, use, lab, val, u0 in zip(ys, S_list, uses, labels, valids, u0_list):
            us = []
            for side in (0, 1):
                v, uh = jax.vmap(lambda yy, SS, uu, ll, vv, u0s, side=side: side_terms(th, side, yy, SS, uu, ll, vv, u0s))(
                    y[:, side], S[:, side], use[:, side], lab[:, side], val[:, side], u0[:, side])
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


def data_vjp(problem: Problem, d, g_S: list[np.ndarray], psi=None) -> tuple[np.ndarray, np.ndarray]:
    r"""(:math:`J^\top g_S`, :math:`(\partial S/\partial\psi)^\top g_S`): gradients of :math:`\sum g_S \cdot S(d, \psi)`, chunked over pings."""
    pb = problem
    f = _vjp_jit(_key(pb))
    d, psi = jnp.asarray(d), _psi(pb, psi)
    grad, grad_psi = np.zeros(pb.mesh.n_vertices), np.zeros((2, 3))
    for poses, g in zip(pb.poses, g_S):
        for idx, padded in _chunks(len(poses), pb.chunk):
            gc = np.zeros((pb.chunk, 2, pb.n_bins))
            gc[: len(idx)] = g[idx]
            gd, gp = f(d, jnp.asarray(poses[padded]), psi, jnp.asarray(gc))
            grad += np.asarray(gd)
            grad_psi += np.asarray(gp)
    return grad, grad_psi


def _args(S, u0, psi):
    return ([jnp.asarray(s) for s in S], None if u0 is None else [jnp.asarray(x) for x in u0], psi)


def value_and_grad(problem: Problem, d, beta, F=None, u0=None, psi=None, S=None):
    r""":math:`\ell(d, \beta, \psi)`, its gradient (d, :math:`\beta`), and the data term's details.

    F: a `data_term(problem)` to reuse (it is jitted once per problem), or a warm one
    (`data_term(problem, warm=True)`) together with u0, the log-kappa modes to start from;
    psi: the sonar parameters (2, 3), default theta's. details hold the rendered S,
    :math:`\hat u` per pass, :math:`\hat\mu` per pair and `grad_psi`, the gradient in :math:`\psi`
    (including its prior when it is free). S: the render at (d, :math:`\psi`) if already done.
    """
    F = data_term(problem) if F is None else F
    psi = _psi(problem, psi)
    S = render(problem, d, psi) if S is None else S
    (val, (u_hats, mus)), (g_S, g_psi_F) = jax.value_and_grad(F, argnums=(0, 2), has_aux=True)(*_args(S, u0, psi))
    grad_d, g_psi_S = data_vjp(problem, d, [np.asarray(g) for g in g_S], psi)
    gd_prior, gb_prior = problem.prior.grad(np.asarray(d), np.asarray(beta))
    lp_psi, g_psi_prior = problem.psi_log_prior(psi)
    value = float(val) + problem.prior.log_density(np.asarray(d), np.asarray(beta)) + lp_psi
    details = {"S": S, "u": [np.asarray(u) for u in u_hats], "mu": [float(m) for m in mus], "data": float(val),
               "grad_psi": np.asarray(g_psi_F) + g_psi_S + g_psi_prior}
    return value, (grad_d + gd_prior, gb_prior), details


def value(problem: Problem, d, beta, F=None, u0=None, psi=None, keep: dict | None = None) -> float:
    r""":math:`\ell(d, \beta, \psi)` alone: one render and the data term, no gradient (warm with u0, as `value_and_grad`).

    keep: a dict that receives the render as "S", to pass on to `value_and_grad` at the same point.
    """
    F = data_term(problem) if F is None else F
    psi = _psi(problem, psi)
    S = render(problem, d, psi)
    if keep is not None:
        keep["S"] = S
    val, _ = F(*_args(S, u0, psi))
    return (float(val) + problem.prior.log_density(np.asarray(d), np.asarray(beta))
            + problem.psi_log_prior(psi)[0])
