r"""Outer loop of inference (paper, Alg. 1), with lit/shadow labels in place of the visibility.

Each round, at the current depths and sonar parameters:

1. label the blocks lit or dark and find each ping-side's first echo, both from the data
   relative to the current gain: labels in the echo's onset zone are left out, and the
   water column before it gives the measured lit rate of blocks without a seabed echo;
   render every ping; refit the spike parameters and the log-:math:`\kappa` prior
   (`bbf.inference.calibrate`) on every `calib_every`-th ping;
2. draw locations in each pair of passes' overlap and select the matched pairs;
3. compute and freeze the per-ping-side Jacobians (and the beam derivatives of :math:`S` when
   the sonar parameters :math:`\psi` are free);
4. ascend the log posterior by Gauss--Newton (`bbf.inference.gauss_newton`), in :math:`\psi` too
   when it is free, and carry the new :math:`\psi` into theta for the next round.

Rounds stop when the matched pairs no longer change or after `rounds`.
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
from bbf.match import draw_overlap, matched_mask, project, select_pairs
from bbf.mesh import locate
from bbf.ping.geometry import trace
from bbf.ping.shadow import block_labels, onset_valid, ping_use, water_column_lit_rate
from bbf.ping.theta import Theta, psi_of, with_psi
from bbf.prior import Prior


def labels_of(ps: Pass, theta: Theta) -> np.ndarray:
    """Lit/shadow labels (n, 2, n_blocks) of every ping-side of a pass."""
    return np.stack([block_labels(ps.y[:, c], theta, c, ps.dr) for c in (0, 1)], axis=1)


def jacobians(problem: Problem, d) -> list[list[list]]:
    """Frozen Jacobians J[p][t][c] of every ping-side at depths d."""
    pb = problem
    out = []
    for poses in pb.poses:
        sides = [ping_side_jacobians(pb.mesh, d, poses, c, pb.theta, pb.u, pb.n_bins, pb.dr, chunk=pb.chunk)
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


def _pair_key(pairs) -> set:
    return {(k, *row) for k, pr in pairs.items()
            for row in zip(pr.ping_a, pr.side_a, pr.bin_a, pr.ping_b, pr.side_b, pr.bin_b)}


def infer(passes: list[Pass], mesh: M.Mesh, prior: Prior, theta: Theta, d0, beta0, u, rounds: int = 3,
          inner: int = 10, n_draws: int = 4000, calib_every: int = 50, chunk: int = 32, seed: int = 0,
          onset: float = 1.5, laplace_samples: int = 0, trend_search=None, psi_prior=None,
          callback=None, log=print) -> dict:
    r"""Run the outer loop from (d0, beta0); returns depths, trend, theta, pairs and history.

    trend_search: in the first round, before Gauss--Newton, try these trends and keep the best
    after refinement (`bbf.inference.search.search_trend`): a dict with `slopes_x`, optionally
    `slopes_y` and `steps`. The mean depth kept per candidate is that of the vertices the passes see.

    psi_prior: (mean, sd), each (2, 3), of the sonar parameters :math:`\psi` (beam centre and width
    in radians, linear gain in dB/m, per side); with it, :math:`\psi` is an unknown of the graph,
    started at theta's values. Otherwise it stays at theta's.

    With `laplace_samples` > 0, also the Laplace posterior standard deviations of the depths
    (`sd_d`) and trend (`sd_beta`) at the final mode, from that many Monte Carlo samples.

    callback(round, d, beta, theta) is called after each round, e.g. to save intermediate results.
    """
    poses = [p.local_poses(mesh.origin) for p in passes]
    ys = [p.y for p in passes]
    dr = passes[0].dr
    block_dr = theta.block * dr
    d, beta = np.asarray(d0, dtype=float), np.asarray(beta0, dtype=float)
    history, prev_pairs, pairs = [], None, {}
    problem = None
    for rnd in range(rounds):
        t0 = time.time()
        # labels and first echoes from the data, with the current gain (it moves when psi is
        # free): the echo's onset zone is left out, and the water column before it gives the
        # lit rate of blocks without a seabed echo, by range
        labels = [labels_of(p, theta) for p in passes]
        r0 = [first_echo(p, theta) for p in passes]
        valid = [onset_valid(r, lab.shape[-1], block_dr, onset) for r, lab in zip(r0, labels)]
        theta = dataclasses.replace(theta, wc_lit=water_column_lit_rate(labels, r0, block_dr, onset))
        base = Problem(mesh=mesh, prior=prior, theta=theta, poses=poses, y=ys, labels=labels, u=u, dr=dr, chunk=chunk,
                       valid=valid)
        S = render(base, d)
        # 1. theta at the current depths, on a subset of pings
        sub = [np.arange(0, len(p), calib_every) for p in poses]
        y_c = np.concatenate([y[s] for y, s in zip(ys, sub)])
        S_c = np.concatenate([Sp[s] for Sp, s in zip(S, sub)])
        use_c = np.concatenate([np.stack([ping_use(lab[t], ys[0].shape[-1], theta.block, valid=val[t]) for t in s])
                                for lab, val, s in zip(labels, valid, sub)])
        theta, fit = calibrate(y_c, S_c, use_c, theta, dr)
        log(f"round {rnd}: theta fit (converged {fit['converged']}): eta {np.asarray(theta.eta)}, "
            f"zeta {np.asarray(theta.zeta)}, ell {np.asarray(theta.ell)}, "
            f"log kappa {np.asarray(theta.log_kappa_mean)} ± {np.asarray(theta.log_kappa_std)}")
        # 2. matched pairs at the current depths
        pairs = {}
        rng = np.random.default_rng(seed + rnd)
        taken = [np.zeros(y.shape, bool) for y in ys]  # each bin in at most one pair, over all pass pairs
        for a in range(len(passes)):
            for b in range(a + 1, len(passes)):
                q = draw_overlap(poses[a], poses[b], ys[a].shape[-1] * dr, n_draws, rng=rng)
                if len(q) == 0:
                    continue
                pa = project(mesh, d, poses[a], q, ys[a].shape[-1], dr)
                pb = project(mesh, d, poses[b], q, ys[b].shape[-1], dr)
                pr = select_pairs(pa, pb, S[a], S[b], ys[a], ys[b], theta, dr, taken_a=taken[a], taken_b=taken[b])
                taken[a] |= matched_mask(pr, len(ys[a]), ys[a].shape[-1], "a")
                taken[b] |= matched_mask(pr, len(ys[b]), ys[b].shape[-1], "b")
                pairs[(a, b)] = pr
        log(f"round {rnd}: {sum(len(p) for p in pairs.values())} matched pairs")
        problem = dataclasses.replace(base, theta=theta, pairs=pairs, psi_prior=psi_prior)
        # 3. frozen Jacobians, 4. Gauss-Newton
        tj = time.time()
        if rnd == 0 and trend_search is not None:
            from bbf.inference.search import search_trend
            seen = observed_vertices(problem, d)
            fixed = dataclasses.replace(problem, psi_prior=None)  # the search moves the trend only
            d, beta, _ = search_trend(fixed, d, beta, observed=seen, log=log, **trend_search)
            log(f"round {rnd}: trend search picked slopes ({beta[0]:+.3f}, {beta[1]:+.3f})")
        J = jacobians(problem, d)
        Jb = beam_jacobian(problem, d) if problem.free_psi else None
        log(f"round {rnd}: Jacobians in {time.time() - tj:.0f}s")
        d, beta, psi, hist = gauss_newton(problem, d, beta, J, Jb=Jb, max_iter=inner, log=log)
        theta = with_psi(theta, psi)
        history.append({"round": rnd, "gn": hist, "n_pairs": sum(len(p) for p in pairs.values()),
                        "time": time.time() - t0})
        if callback is not None:
            callback(rnd, d, beta, theta)
        key = _pair_key(pairs)
        if prev_pairs is not None and key == prev_pairs:
            log(f"round {rnd}: matched pairs unchanged; done")
            break
        prev_pairs = key
    if problem is not None:
        problem = dataclasses.replace(problem, theta=theta)  # with the final psi
    out = {"d": d, "beta": beta, "theta": theta, "psi": psi_of(theta), "pairs": pairs, "history": history,
           "problem": problem}
    if laplace_samples > 0 and problem is not None:
        t0 = time.time()
        Jb = beam_jacobian(problem, d) if problem.free_psi else None
        lap = laplace(problem, d, beta, jacobians(problem, d), n_samples=laplace_samples, rng=seed, Jb=Jb)
        out.update(sd_d=lap["sd_d"], sd_beta=lap["sd_beta"])
        if "sd_psi" in lap:
            out["sd_psi"] = lap["sd_psi"]
        log(f"Laplace: depth sd median {np.median(lap['sd_d']):.2f} m in {time.time() - t0:.0f}s")
    return out
