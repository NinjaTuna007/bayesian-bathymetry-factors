r"""Outer loop of inference (paper, Alg. 1), with lit/shadow labels in place of the visibility.

Labels depend on the data only and are computed once, together with each ping-side's first
echo: labels in the echo's onset zone are left out, and the water column before it gives
the measured lit rate of blocks without a seabed echo. Each round, at the current depths:

1. render every ping; refit the spike parameters and the log-:math:`\kappa` prior
   (`bbf.inference.calibrate`) on every `calib_every`-th ping;
2. draw locations in each pair of passes' overlap and select the matched pairs;
3. compute and freeze the per-ping-side Jacobians;
4. ascend the log posterior by Gauss--Newton (`bbf.inference.gauss_newton`).

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
from bbf.inference.jacobian import ping_side_jacobian
from bbf.inference.objective import Problem, render
from bbf.io import Pass
from bbf.match import draw_overlap, project, select_pairs
from bbf.ping.shadow import block_labels, onset_valid, ping_use, water_column_lit_rate
from bbf.ping.theta import Theta
from bbf.prior import Prior


def labels_of(ps: Pass, theta: Theta) -> np.ndarray:
    """Lit/shadow labels (n, 2, n_blocks) of every ping-side of a pass."""
    return np.stack([[block_labels(ps.y[t, c], theta, c, ps.dr) for c in (0, 1)] for t in range(ps.n_pings)])


def jacobians(problem: Problem, d) -> list[list[list]]:
    """Frozen Jacobians J[p][t][c] of every ping-side at depths d."""
    d = jnp.asarray(d)
    return [[[ping_side_jacobian(problem.mesh, d, jnp.asarray(pose), c, problem.theta, problem.u, problem.n_bins,
                                 problem.dr) for c in (0, 1)] for pose in poses] for poses in problem.poses]


def _pair_key(pairs) -> set:
    return {(k, *row) for k, pr in pairs.items()
            for row in zip(pr.ping_a, pr.side_a, pr.bin_a, pr.ping_b, pr.side_b, pr.bin_b)}


def infer(passes: list[Pass], mesh: M.Mesh, prior: Prior, theta: Theta, d0, beta0, u, rounds: int = 3,
          inner: int = 10, n_draws: int = 4000, calib_every: int = 50, chunk: int = 32, seed: int = 0,
          onset: float = 1.5, callback=None, log=print) -> dict:
    """Run the outer loop from (d0, beta0); returns depths, trend, theta, pairs and history.

    callback(round, d, beta, theta) is called after each round, e.g. to save intermediate results.
    """
    poses = [p.local_poses(mesh.origin) for p in passes]
    ys = [p.y for p in passes]
    dr = passes[0].dr
    labels = [labels_of(p, theta) for p in passes]
    # first echoes from the data: the echo's onset zone is left out, and the water column
    # before it gives the lit rate of blocks without a seabed echo, by range
    r0 = [first_echo(p, theta) for p in passes]
    block_dr = theta.block * dr
    valid = [onset_valid(r, lab.shape[-1], block_dr, onset) for r, lab in zip(r0, labels)]
    theta = dataclasses.replace(theta, wc_lit=water_column_lit_rate(labels, r0, block_dr, onset))
    d, beta = np.asarray(d0, dtype=float), np.asarray(beta0, dtype=float)
    history, prev_pairs, pairs = [], None, {}
    problem = None
    for rnd in range(rounds):
        t0 = time.time()
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
        for a in range(len(passes)):
            for b in range(a + 1, len(passes)):
                q = draw_overlap(poses[a], poses[b], ys[a].shape[-1] * dr, n_draws, rng=rng)
                if len(q) == 0:
                    continue
                pa = project(mesh, d, poses[a], q, ys[a].shape[-1], dr)
                pb = project(mesh, d, poses[b], q, ys[b].shape[-1], dr)
                pairs[(a, b)] = select_pairs(pa, pb, S[a], S[b], ys[a], ys[b], theta, dr)
        log(f"round {rnd}: {sum(len(p) for p in pairs.values())} matched pairs")
        problem = dataclasses.replace(base, theta=theta, pairs=pairs)
        # 3. frozen Jacobians, 4. Gauss-Newton
        tj = time.time()
        J = jacobians(problem, d)
        log(f"round {rnd}: Jacobians in {time.time() - tj:.0f}s")
        d, beta, hist = gauss_newton(problem, d, beta, J, max_iter=inner, log=log)
        history.append({"round": rnd, "gn": hist, "n_pairs": sum(len(p) for p in pairs.values()),
                        "time": time.time() - t0})
        if callback is not None:
            callback(rnd, d, beta, theta)
        key = _pair_key(pairs)
        if prev_pairs is not None and key == prev_pairs:
            log(f"round {rnd}: matched pairs unchanged; done")
            break
        prev_pairs = key
    return {"d": d, "beta": beta, "theta": theta, "pairs": pairs, "history": history, "problem": problem}
