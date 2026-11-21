r"""Outer loop of inference (paper, Alg. 1), with lit/shadow labels in place of the visibility.

Each round, at the current depths, reflectivity and sonar parameters:

1. label the blocks lit or dark and find each ping-side's first echo, both from the data
   relative to the current gain: labels in the echo's onset zone are left out, and the
   water column before it gives the measured lit rate of blocks without a seabed echo;
   render every ping; refit the spike parameters and the log-:math:`\kappa` prior
   (`bbf.inference.calibrate`) on every `calib_every`-th ping;
2. compute and freeze the per-ping-side Jacobians (and the beam derivatives of :math:`S` when
   the sonar parameters :math:`\psi` are free);
3. ascend the log posterior by Gauss--Newton (`bbf.inference.gauss_newton`), in the
   log-reflectivity field :math:`\xi` when the problem has one (`refl`) and in :math:`\psi` when it
   is free, and carry the new :math:`\psi` into theta for the next round.

The passes are matched through :math:`\xi` (`bbf.match`): they render through one field, so
nothing is paired or registered between rounds. Before the first round's ascent, :math:`\xi` is
fitted with the depths held at the start: stepping both from :math:`\xi = 0` lets the depths take
up shading that is reflectivity, into rough local optima they do not leave.
"""
import dataclasses
import time

import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.inference.calibrate import calibrate
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.init import first_echo
from bbf.inference.jacobian import ping_side_jacobians
from bbf.inference.laplace import laplace
from bbf.inference.objective import Problem, beam_jacobian, render
from bbf.io import Pass
from bbf.match import Reflectivity
from bbf.mesh import locate
from bbf.ping.first_return import FirstReturn
from bbf.ping.geometry import trace
from bbf.ping.shadow import block_labels, onset_valid, ping_use, water_column_lit_rate
from bbf.ping.theta import Theta, psi_of, with_psi
from bbf.prior import Prior


def labels_of(ps: Pass, theta: Theta) -> np.ndarray:
    """Lit/shadow labels (n, 2, n_blocks) of every ping-side of a pass."""
    return np.stack([block_labels(ps.y[:, c], theta, c, ps.dr, ps.log_power()) for c in (0, 1)], axis=1)


def beyond(n_blocks: int, block_dr: float, min_range: float) -> np.ndarray:
    """Blocks (n_blocks,) whose centre is at least `min_range` m from the sonar."""
    return (np.arange(n_blocks) + 0.5) * block_dr >= min_range


def jacobians(problem: Problem, d, xi=None) -> list[list[list]]:
    r"""Frozen Jacobians J[p][t][c] of every ping-side at depths d; with the field, in :math:`(d, \xi)` at xi (default 0)."""
    pb = problem
    if pb.refl is not None and xi is None:
        xi = np.zeros(pb.mesh.n_vertices)
    out = []
    for poses in pb.poses:
        sides = [ping_side_jacobians(pb.mesh, d, poses, c, pb.theta, pb.u, pb.n_bins, pb.dr, chunk=pb.chunk,
                                     xi=xi if pb.refl is not None else None)
                 for c in (0, 1)]
        out.append([[sides[0][t], sides[1][t]] for t in range(len(poses))])
    return out


def observed_vertices(problem: Problem, d) -> np.ndarray:
    """Vertices (n,) touched by some ping-side's curve at depths d."""
    mesh = problem.mesh
    seen = np.zeros(mesh.n_vertices, bool)
    for poses in problem.poses:
        for pose in poses[:: max(len(poses) // 50, 1)]:
            for c in (0, 1):
                q = np.asarray(trace(mesh, jnp.asarray(d), jnp.asarray(pose), c, problem.u))
                tri, _, inside = (np.asarray(a) for a in locate(mesh, q))
                seen[mesh.triangles[tri[inside]].ravel()] = True
    return seen


def infer(passes: list[Pass], mesh: M.Mesh, prior: Prior, theta: Theta, d0, beta0, u, rounds: int = 3,
          inner: int = 10, calib_every: int = 50, chunk: int = 32, seed: int = 0,
          onset: float = 1.5, laplace_samples: int = 0, trend_search=None, psi_prior=None,
          refl: Reflectivity | None = None, xi0=None, first_return: FirstReturn | None = None, callback=None,
          min_range: float = 0.0, fit_shape: bool = False, log=print) -> dict:
    r"""Run the outer loop from (d0, beta0); returns depths, trend, reflectivity, theta and history.

    refl: the prior of the log-reflectivity field :math:`\xi`; with it, :math:`\xi` is an unknown of
    the graph shared by all passes, started at xi0 (default: fitted at d0) and returned as `xi`. Without it
    each ping-side has its one reflectivity :math:`\kappa_{t,c}` and the passes are not matched.

    trend_search: in the first round, before Gauss--Newton, try these trends and keep the best
    after refinement (`bbf.inference.search.search_trend`): a dict with `slopes_x`, optionally
    `slopes_y` and `steps`. The mean depth kept per candidate is that of the vertices the passes see.

    psi_prior: (mean, sd), each (2, 3), of the sonar parameters :math:`\psi` (beam centre and width
    in radians, linear gain in dB/m, per side); with it, :math:`\psi` is an unknown of the graph,
    started at theta's values. Otherwise it stays at theta's.

    With `laplace_samples` > 0, also the Laplace posterior standard deviations of the depths
    (`sd_d`), trend (`sd_beta`) and field (`sd_xi`) at the final mode, from that many Monte Carlo samples.

    first_return: the first echoes and the first-return factor's constants
    (`bbf.ping.first_return`); it ties the seabed under the tracks, where the beam predicts little
    energy, to the measured range. Its r0 must come from these passes.

    min_range: blocks nearer than this (m) are left out of both the ping and the shadow factor,
    e.g. the sonar's own ringing and sensor-fixed lines in the near water column, which otherwise
    set the spike parameters; it must stay short of the nearest seabed.

    fit_shape: refit the gamma shape and the spike rates with the spike parameters each round
    (`bbf.inference.calibrate`), instead of keeping theta's.

    callback(round, d, beta, theta) is called after each round, e.g. to save intermediate results.
    """
    poses = [p.local_poses(mesh.origin) for p in passes]
    ys = [p.y for p in passes]
    dr = passes[0].dr
    block_dr = theta.block * dr
    d, beta = np.asarray(d0, dtype=float), np.asarray(beta0, dtype=float)
    xi = np.zeros(mesh.n_vertices) if xi0 is None else np.asarray(xi0, dtype=float)
    offsets = [p.log_power() for p in passes]  # transmit power: a known per-ping gain
    history = []
    problem = None
    for rnd in range(rounds):
        t0 = time.time()
        # labels and first echoes from the data, with the current gain (it moves when psi is
        # free): the echo's onset zone is left out, and the water column before it gives the
        # lit rate of blocks without a seabed echo, by range
        labels = [labels_of(p, theta) for p in passes]
        r0 = [first_echo(p, theta) for p in passes]
        valid = [onset_valid(r, lab.shape[-1], block_dr, onset) & beyond(lab.shape[-1], block_dr, min_range)
                 for r, lab in zip(r0, labels)]
        theta = dataclasses.replace(theta, wc_lit=water_column_lit_rate(labels, r0, block_dr, onset))
        base = Problem(mesh=mesh, prior=prior, theta=theta, poses=poses, y=ys, labels=labels, u=u, dr=dr, chunk=chunk,
                       valid=valid, refl=refl, kappa_offset=offsets,
                       first_return=first_return)
        S = render(base, d, xi=xi)
        # 1. theta at the current depths, on a subset of pings
        sub = [np.arange(0, len(p), calib_every) for p in poses]
        y_c = np.concatenate([y[s] for y, s in zip(ys, sub)])
        S_c = np.concatenate([Sp[s] for Sp, s in zip(S, sub)])
        use_c = np.concatenate([np.stack([ping_use(lab[t], ys[0].shape[-1], theta.block, valid=val[t]) for t in s])
                                for lab, val, s in zip(labels, valid, sub)])
        theta, fit = calibrate(y_c, S_c, use_c, theta, dr,
                               offset=np.concatenate([o[s] for o, s in zip(offsets, sub)]), fit_shape=fit_shape)
        if fit_shape:
            log(f"round {rnd}: shape L {theta.L:.2f}, spike rates {np.asarray(theta.eps).round(3)}")
        log(f"round {rnd}: theta fit (converged {fit['converged']}): eta {np.asarray(theta.eta)}, "
            f"zeta {np.asarray(theta.zeta)}, ell {np.asarray(theta.ell)}, "
            f"log kappa {np.asarray(theta.log_kappa_mean)} ± {np.asarray(theta.log_kappa_std)}")
        problem = dataclasses.replace(base, theta=theta, psi_prior=psi_prior)
        if rnd == 0 and refl is not None and xi0 is None:
            fixed = dataclasses.replace(problem, psi_prior=None)
            _, _, _, xi, _ = gauss_newton(fixed, d, beta, jacobians(fixed, d, xi), xi=xi, hold_depths=True,
                                          max_iter=inner, log=log)
            log(f"round {rnd}: reflectivity at the start depths, sd {xi.std():.2f}")
        # 2. frozen Jacobians, 3. Gauss-Newton
        tj = time.time()
        if rnd == 0 and trend_search is not None:
            from bbf.inference.search import search_trend
            seen = observed_vertices(problem, d)
            fixed = dataclasses.replace(problem, psi_prior=None)  # the search moves the trend only
            d, beta, xi, _ = search_trend(fixed, d, beta, observed=seen, xi=xi, log=log, **trend_search)
            log(f"round {rnd}: trend search picked slopes ({beta[0]:+.3f}, {beta[1]:+.3f})")
        J = jacobians(problem, d, xi)
        Jb = beam_jacobian(problem, d, xi=xi) if problem.free_psi else None
        log(f"round {rnd}: Jacobians in {time.time() - tj:.0f}s")
        d, beta, psi, xi, hist = gauss_newton(problem, d, beta, J, Jb=Jb, xi=xi, max_iter=inner, log=log)
        theta = with_psi(theta, psi)
        history.append({"round": rnd, "gn": hist, "time": time.time() - t0})
        if callback is not None:
            callback(rnd, d, beta, theta)
    if problem is not None:
        problem = dataclasses.replace(problem, theta=theta)  # with the final psi
    out = {"d": d, "beta": beta, "xi": xi, "theta": theta, "psi": psi_of(theta), "history": history,
           "problem": problem}
    if laplace_samples > 0 and problem is not None:
        t0 = time.time()
        Jb = beam_jacobian(problem, d, xi=xi) if problem.free_psi else None
        lap = laplace(problem, d, beta, jacobians(problem, d, xi), n_samples=laplace_samples, rng=seed, Jb=Jb, xi=xi)
        out.update({k: lap[k] for k in ("sd_d", "sd_beta", "sd_psi", "sd_xi") if k in lap})
        log(f"Laplace: depth sd median {np.median(lap['sd_d']):.2f} m in {time.time() - t0:.0f}s")
    return out
