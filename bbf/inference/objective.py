r"""The log posterior of the depths and its gradient, through the rendered unit intensities.

Every data factor depends on the depths only through the unit intensities :math:`S` of the
pings (`bbf.ping.unit_intensity`): the ping factor through each ping-side's :math:`S` and the
shadow factor through block means of :math:`S`. So

.. math::

    \ell(d, \beta) = \log p(\beta) + \log p(d \mid \beta) + F\bigl(S(d)\bigr),

and :math:`\nabla_d F = J^\top \nabla_S F` with :math:`J = \partial S / \partial d`. The first-return
and horizon factors (`Problem.first_return`, `Problem.horizon`) depend on the geometry alone and
are added with their own gradients; with the horizon factor the dark-block shadow term is left out. :math:`\nabla_S F`
is one JAX gradient over all pings; :math:`J^\top` is applied in chunks of pings, so memory
stays that of one chunk.

With `Problem.refl` set, the log-reflectivity field :math:`\xi` on the vertices (`bbf.match`) is
an unknown too. It enters through the render alone, :math:`S = S(d, \xi)`, which is how the
passes are matched: all of them render through the same field. Its prior :math:`\log p(\xi)` is
added, and the gradient in :math:`\xi` is :math:`(\partial S / \partial \xi)^\top \nabla_S F` from the same
reverse pass. Without it :math:`\xi = 0`.

The sonar parameters :math:`\psi` (beam centre and width, linear gain; `bbf.ping.theta.psi_of`)
are an argument too: the beam reaches :math:`F` through :math:`S`, the gain directly. With
`Problem.psi_prior` set they are unknowns of the graph with a Gaussian prior, and the gradient
includes them; otherwise they stay at the values in `Problem.theta`. `Problem.psi_held` keeps
some of them fixed while the others are fitted, e.g. the gain with the beam free: the labels and
the background are relative to the gain, so a fitted gain chooses which bins are scored.
"""
from dataclasses import dataclass
from functools import cache, cached_property, lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.match.reflectivity import Reflectivity
from bbf.ping.factor import WARM_ITERS, bin_ranges, laplace_ping_side, log_gain
from bbf.ping.first_return import FirstReturn, first_return_term
from bbf.ping.horizon import Horizon, horizon_term
from bbf.ping.render import unit_intensity
from bbf.ping.shadow import log_shadow_side, ping_use
from bbf.ping.theta import Theta, psi_of, render_key, render_theta, with_psi
from bbf.prior import Prior


@dataclass(frozen=True, eq=False)  # hashed by identity, so the data term can be cached per problem
class Problem:
    """Everything the objective needs, with the outer loop's choices (labels) fixed.

    poses: per pass (n, 6), mesh-local; y: per pass (n, 2, B) intensities; labels: per pass
    (n, 2, n_blocks) lit/shadow labels.
    """
    mesh: M.Mesh
    prior: Prior
    theta: Theta
    poses: list[np.ndarray]
    y: list[np.ndarray]
    labels: list[np.ndarray]
    u: np.ndarray
    dr: float
    chunk: int = 32
    valid: list[np.ndarray] | None = None  # per pass (n, 2, n_blocks): blocks to use (see onset_valid)
    psi_prior: tuple[np.ndarray, np.ndarray] | None = None  # (mean, sd) of psi (2, 3); None: psi fixed
    psi_held: np.ndarray | None = None  # (2, 3) bool: components of a free psi kept at their start; None: none
    refl: Reflectivity | None = None  # prior of the log-reflectivity field xi; None: no field
    kappa_offset: list[np.ndarray] | None = None  # per pass (n,): known shift of the log-kappa prior mean (log power)
    first_return: FirstReturn | None = None  # first echoes per ping-side; None: no first-return factor
    horizon: Horizon | None = None  # horizon factor on the labels; None: the dark-block shadow term instead
    temper: float = 1.0  # factor on the intensity log likelihood (ping and shadow factors): bins are not independent
    mode_grid: int = 0  # starts of each log-kappa mode search (`bbf.ping.factor.mode`); 0: from one start
    shadow_stop: bool = True  # hold log-kappa fixed in the shadow factor's derivative (`log_shadow_side`)
    shadow_gain: bool = False  # score dark blocks against the gain the labels were taken with (`lit_probability`)

    @property
    def free_psi(self) -> bool:
        return self.psi_prior is not None

    @property
    def n_xi(self) -> int:
        r"""Number of unknowns of :math:`\xi`: one per vertex, or none without the field."""
        return 0 if self.refl is None else self.mesh.n_vertices

    def psi_log_prior(self, psi) -> tuple[float, np.ndarray]:
        r"""Gaussian log prior of :math:`\psi` and its gradient (2, 3); zero when :math:`\psi` is fixed."""
        if self.psi_prior is None:
            return 0.0, np.zeros((2, 3))
        mean, sd = (np.asarray(a, float) for a in self.psi_prior)
        z = (np.asarray(psi) - mean) / sd
        return float(-0.5 * np.sum(z**2) - np.sum(np.log(sd)) - z.size * 0.5 * np.log(2 * np.pi)), -z / sd

    def xi_log_prior(self, xi) -> tuple[float, np.ndarray]:
        r""":math:`\log p(\xi)` and its gradient (n,); zero without the field."""
        if self.refl is None:
            return 0.0, np.zeros(self.mesh.n_vertices)
        return self.refl.log_density(np.asarray(xi)), self.refl.grad(np.asarray(xi))

    @property
    def n_bins(self) -> int:
        return self.y[0].shape[-1]

    def uses(self) -> list[np.ndarray]:
        """Per pass (n, 2, B): bins the ping factor scores (those in valid lit blocks)."""
        return self._uses

    @cached_property
    def _uses(self) -> list[np.ndarray]:
        out = []
        for p, (y, lab) in enumerate(zip(self.y, self.labels, strict=True)):
            val = None if self.valid is None else self.valid[p]
            out.append(np.stack([ping_use(lab[t], self.n_bins, self.theta.block, None if val is None else val[t])
                                 for t in range(len(y))]))
        return out


def _chunks(n: int, size: int):
    """Index arrays of fixed length `size` covering 0..n-1; the last is padded with its final index."""
    for s in range(0, n, size):
        idx = np.arange(s, min(s + size, n))
        yield idx, np.concatenate([idx, np.full(size - len(idx), idx[-1])])


def _key(pb: Problem) -> tuple:
    """What the compiled render depends on: the mesh, theta's render fields, the samples and bins."""
    return (pb.mesh, render_key(pb.theta), np.asarray(pb.u).tobytes(), pb.n_bins, pb.dr)


def _render_chunk(key: tuple, d, poses, psi, xi):
    mesh, rkey, u_bytes, n_bins, dr = key
    th = render_theta(rkey, psi)
    u = np.frombuffer(u_bytes)
    return jnp.stack([jax.vmap(lambda pose, side=side: unit_intensity(mesh, d, pose, side, th, u, n_bins, dr, xi))(poses)
                      for side in (0, 1)], axis=1)


@cache
def _render_jit(key: tuple):
    return jax.jit(lambda d, poses, psi, xi: _render_chunk(key, d, poses, psi, xi))


@cache
def _vjp_jit(key: tuple):
    def chunk_vjp(d, poses, psi, xi, g):
        _, vjp = jax.vjp(lambda d, psi, xi: _render_chunk(key, d, poses, psi, xi), d, psi, xi)
        return vjp(g)
    return jax.jit(chunk_vjp)


@cache
def _beam_jvp_jit(key: tuple):
    def chunk_jvp(d, poses, psi, xi):
        # side c's S depends only on side c's beam, so one tangent per column covers both sides
        cols = []
        for k in (0, 1):
            t = jnp.zeros_like(psi).at[:, k].set(1.0)
            cols.append(jax.jvp(lambda p: _render_chunk(key, d, poses, p, xi), (psi,), (t,))[1])
        return jnp.stack(cols, axis=-1)
    return jax.jit(chunk_jvp)


def _psi(problem: Problem, psi):
    return jnp.asarray(psi_of(problem.theta) if psi is None else psi, dtype=float)


def _xi(problem: Problem, xi):
    r""":math:`\xi` as the render takes it: zeros when not given (the compiled render always has the argument)."""
    return jnp.zeros(problem.mesh.n_vertices) if xi is None else jnp.asarray(xi, dtype=float)


def render(problem: Problem, d, psi=None, xi=None) -> list[np.ndarray]:
    r"""Unit intensities (n, 2, B) of every ping of every pass at depths d (:math:`\psi` default theta's, :math:`\xi` default 0)."""
    pb = problem
    f = _render_jit(_key(pb))
    d, psi, xi = jnp.asarray(d), _psi(pb, psi), _xi(pb, xi)
    out = []
    for poses in pb.poses:
        S = np.zeros((len(poses), 2, pb.n_bins))
        for idx, padded in _chunks(len(poses), pb.chunk):
            S[idx] = np.asarray(f(d, jnp.asarray(poses[padded]), psi, xi))[: len(idx)]
        out.append(S)
    return out


def beam_jacobian(problem: Problem, d, psi=None, xi=None) -> list[np.ndarray]:
    r"""Per pass (n, 2, B, 2): :math:`\partial S / \partial(\beta_0, \omega)` of each ping-side, with respect to its own side's beam."""
    pb = problem
    f = _beam_jvp_jit(_key(pb))
    d, psi, xi = jnp.asarray(d), _psi(pb, psi), _xi(pb, xi)
    out = []
    for poses in pb.poses:
        Jb = np.zeros((len(poses), 2, pb.n_bins, 2))
        for idx, padded in _chunks(len(poses), pb.chunk):
            Jb[idx] = np.asarray(f(d, jnp.asarray(poses[padded]), psi, xi))[: len(idx)]
        out.append(Jb)
    return out


@lru_cache(maxsize=8)  # compiled once per problem (hashed by identity); a round uses a few
def data_term(problem: Problem, warm: bool = False):
    r"""Jitted :math:`F(S, \hat u_0, \psi)` over all passes: (value, :math:`\hat u` per pass (n, 2)).

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
    offsets = ([jnp.zeros(y.shape[0]) for y in ys] if pb.kappa_offset is None
               else [jnp.asarray(o, dtype=float) for o in pb.kappa_offset])

    def side_terms(th, side, y, S, use, lab, val, u0, off):
        if warm:
            ping, u_hat = laplace_ping_side(th, side, dr, y, S, use, u0=u0, iters=WARM_ITERS, offset=off,
                                            temper=pb.temper, grid=pb.mode_grid)
        else:
            ping, u_hat = laplace_ping_side(th, side, dr, y, S, use, offset=off, temper=pb.temper, grid=pb.mode_grid)
        if pb.horizon is not None:  # the horizon factor scores the labels instead
            return ping, u_hat
        ratio: jax.Array | float = 1.0
        if pb.shadow_gain:  # the labels were thresholded with the gain of `pb.theta`
            B, r = pb.theta.block, bin_ranges(pb.n_bins, dr)
            lg = log_gain(r, th.at(side).gain_db) - log_gain(r, pb.theta.at(side).gain_db)
            ratio = jnp.exp(lg[: pb.n_bins // B * B].reshape(-1, B).mean(axis=1))
        return ping + pb.temper * log_shadow_side(th, side, lab, S, u_hat, val, dr, off, stop=pb.shadow_stop,
                                                  gain_ratio=ratio), u_hat

    psi0 = _psi(pb, None)

    def F(S_list, u0_list=None, psi=None):
        th = with_psi(pb.theta, psi0 if psi is None else psi)
        total, u_hats = 0.0, []
        if u0_list is None:
            u0_list = [jnp.zeros(y.shape[:2]) for y in ys]
        for y, S, use, lab, val, u0, off in zip(ys, S_list, uses, labels, valids, u0_list, offsets):
            us = []
            for side in (0, 1):
                v, uh = jax.vmap(lambda yy, SS, uu, ll, vv, u0s, oo, side=side:
                                 side_terms(th, side, yy, SS, uu, ll, vv, u0s, oo))(
                    y[:, side], S[:, side], use[:, side], lab[:, side], val[:, side], u0[:, side], off)
                total = total + v.sum()
                us.append(uh)
            u_hats.append(jnp.stack(us, axis=1))
        return total, u_hats

    return jax.jit(F)


def data_vjp(problem: Problem, d, g_S: list[np.ndarray], psi=None, xi=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Gradients of :math:`\sum g_S \cdot S(d, \psi, \xi)` in d (n,), :math:`\psi` (2, 3) and :math:`\xi` (n,), chunked over pings."""
    pb = problem
    f = _vjp_jit(_key(pb))
    d, psi, xi = jnp.asarray(d), _psi(pb, psi), _xi(pb, xi)
    grad, grad_psi, grad_xi = np.zeros(pb.mesh.n_vertices), np.zeros((2, 3)), np.zeros(pb.mesh.n_vertices)
    for poses, g in zip(pb.poses, g_S):
        for idx, padded in _chunks(len(poses), pb.chunk):
            gc = np.zeros((pb.chunk, 2, pb.n_bins))
            gc[: len(idx)] = g[idx]
            gd, gp, gx = f(d, jnp.asarray(poses[padded]), psi, xi, jnp.asarray(gc))
            grad += np.asarray(gd)
            grad_psi += np.asarray(gp)
            grad_xi += np.asarray(gx)
    return grad, grad_psi, grad_xi


def _args(S, u0, psi):
    return ([jnp.asarray(s) for s in S], None if u0 is None else [jnp.asarray(x) for x in u0], psi)


def _first_return(problem: Problem, d):
    """(value, gradient in d, (GGN rows A, weights W)) of the first-return factor; zeros and None without it."""
    if problem.first_return is None:
        return 0.0, np.zeros(problem.mesh.n_vertices), None
    val, grad, A, W = first_return_term(problem.mesh, d, problem.poses, problem.first_return)
    return val, grad, (A, W)


def _horizon(problem: Problem, d, rows: bool = True):
    """(value, gradient in d, (GGN rows A, weights W)) of the horizon factor; zeros and None without it."""
    pb = problem
    if pb.horizon is None:
        return 0.0, np.zeros(pb.mesh.n_vertices), None
    val, grad, A, W = horizon_term(pb.mesh, d, pb.poses, pb.labels, pb.theta, pb.u, pb.dr, pb.horizon, rows=rows,
                                   chunk=pb.chunk)
    return val, grad, None if A is None else (A, W)


def value_and_grad(problem: Problem, d, beta, F=None, u0=None, psi=None, S=None, xi=None):
    r""":math:`\ell(d, \beta, \psi, \xi)`, its gradient (d, :math:`\beta`), and the data term's details.

    F: a `data_term(problem)` to reuse (it is jitted once per problem), or a warm one
    (`data_term(problem, warm=True)`) together with u0, the log-kappa modes to start from;
    psi: the sonar parameters (2, 3), default theta's; xi: the log-reflectivity field (n,),
    default 0. details hold the rendered S, :math:`\hat u` per pass, the data term's value and its
    gradient in d (`grad_d_data`), and
    `grad_psi` and `grad_xi`, the gradients in :math:`\psi` and :math:`\xi` (including their priors
    when they are unknowns), and `first_return` and `horizon`, those factors' GGN rows and weights
    (`ggn.assemble`) or None. S: the render at (d, :math:`\psi`, :math:`\xi`) if already done.
    """
    F = data_term(problem) if F is None else F
    psi = _psi(problem, psi)
    S = render(problem, d, psi, xi) if S is None else S
    (val, u_hats), (g_S, g_psi_F) = jax.value_and_grad(F, argnums=(0, 2), has_aux=True)(*_args(S, u0, psi))
    grad_d, g_psi_S, g_xi_S = data_vjp(problem, d, [np.asarray(g) for g in g_S], psi, xi)
    gd_prior, gb_prior = problem.prior.grad(np.asarray(d), np.asarray(beta))
    lp_psi, g_psi_prior = problem.psi_log_prior(psi)
    lp_xi, g_xi_prior = problem.xi_log_prior(_xi(problem, xi))
    lp_fr, g_fr, fr = _first_return(problem, d)
    lp_hz, g_hz, hz = _horizon(problem, d)
    value = float(val) + problem.prior.log_density(np.asarray(d), np.asarray(beta)) + lp_psi + lp_xi + lp_fr + lp_hz
    details = {"S": S, "u": [np.asarray(u) for u in u_hats], "data": float(val), "grad_d_data": grad_d,
               "grad_psi": np.asarray(g_psi_F) + g_psi_S + g_psi_prior, "grad_xi": g_xi_S + g_xi_prior,
               "first_return": fr, "first_return_value": lp_fr, "horizon": hz, "horizon_value": lp_hz}
    return value, (grad_d + gd_prior + g_fr + g_hz, gb_prior), details


def value(problem: Problem, d, beta, F=None, u0=None, psi=None, keep: dict | None = None, xi=None) -> float:
    r""":math:`\ell(d, \beta, \psi, \xi)` alone: one render and the data term, no gradient (warm with u0, as `value_and_grad`).

    keep: a dict that receives the render as "S", to pass on to `value_and_grad` at the same point.
    """
    F = data_term(problem) if F is None else F
    psi = _psi(problem, psi)
    S = render(problem, d, psi, xi)
    if keep is not None:
        keep["S"] = S
    val, _ = F(*_args(S, u0, psi))
    return (float(val) + problem.prior.log_density(np.asarray(d), np.asarray(beta))
            + problem.psi_log_prior(psi)[0] + problem.xi_log_prior(_xi(problem, xi))[0]
            + _first_return(problem, d)[0] + _horizon(problem, d, rows=False)[0])
