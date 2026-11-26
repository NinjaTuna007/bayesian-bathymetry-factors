r"""Outer loop of inference (paper, Alg. 1), with lit/shadow labels in place of the visibility.

Each round, at the current depths, reflectivity and sonar parameters:

1. find each ping-side's first echo, measure the noise floor by range from the water column
   before it, and label the blocks lit or dark against that floor (not against the model's
   gain, which moves with :math:`\psi`): labels in the echo's onset zone are left out, and the
   water column gives the measured lit rate of blocks without a seabed echo;
   render every ping; refit the spike parameters and the log-:math:`\kappa` prior
   (`bbf.inference.calibrate`) on every `calib_every`-th ping. With `fit_shape`, the first round
   also fits the gamma shape and the spike rates of this level's bins, once the field has had its
   first fit, and takes the labels again at the fitted shape;
2. compute the per-ping-side Jacobians (and the beam derivatives of :math:`S` when
   the sonar parameters :math:`\psi` are free);
3. ascend the log posterior by Gauss--Newton (`bbf.inference.gauss_newton`), with the Jacobians
   taken again after every step, in the
   log-reflectivity field :math:`\xi` when the problem has one (`refl`) and in :math:`\psi` when it
   is free, and carry the new :math:`\psi` into theta for the next round.

The passes are matched through :math:`\xi` (`bbf.match`): they render through one field, so
nothing is paired or registered between rounds. Before the first round's ascent, :math:`\xi` is
fitted with the depths held at the start: stepping both from :math:`\xi = 0` lets the depths take
up shading that is reflectivity, into rough local optima they do not leave.

Or they are matched by the ratio match (`bbf.match.ratio`, `with_ratio`): each round, once theta is
final for it, the scored bins are put in ground cells and visits at the current depths, and the
ping-sides' gains and the visits' shapes are taken there; all of it is held through the round's
Gauss--Newton steps, while the Jacobians are retaken after every step.
"""
import dataclasses
import time

import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.inference.calibrate import calibrate
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.init import far_edge as find_far_edge
from bbf.inference.init import first_echo
from bbf.inference.jacobian import ping_side_jacobians
from bbf.inference.laplace import laplace
from bbf.inference.objective import (
    Problem,
    beam_jacobian,
    ratio_gains,
    ratio_means,
    render,
)
from bbf.io import Pass
from bbf.match import Reflectivity
from bbf.match.cells import footprints
from bbf.match.ratio import RatioMatch, assign, visit_shapes
from bbf.mesh import locate
from bbf.ping.first_return import FirstReturn
from bbf.ping.geometry import trace
from bbf.ping.horizon import Horizon, scored_blocks
from bbf.ping.shadow import (
    block_labels,
    measured_floor,
    onset_valid,
    ping_use,
    water_column_lit_rate,
)
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


def with_ratio(problem: Problem, d, times: list[np.ndarray], rm: RatioMatch, log=print) -> Problem:
    r"""The problem with the ratio match (`bbf.match.ratio`) at depths d: cells and visits, gains and shapes, all held.

    times: per pass (n,), the ping times that split a cell's bins of one pass into visits. Each scored
    bin (`Problem.uses_all`: valid lit blocks, past `min_range` and outside the echo's onset zone) is put
    in the ground cell of its footprint at d (`bbf.match.cells.footprints`, at theta's beam); bins
    without a footprint (no seabed echo) stay in the ping factor. Then, at the render at d, each
    ping-side's gain (`objective.ratio_gains`) and each visit's shape with the model error
    (`visit_shapes`), which the round holds: followed, the gains give gradients with no curvature
    behind them and the shapes let the factor widen itself.
    """
    pb = problem
    if pb.refl is not None:
        raise ValueError("the ratio match replaces the reflectivity field: give one of ratio and refl")
    t0 = time.time()
    fp = [footprints(pb.mesh, jnp.asarray(d), P, pb.theta, pb.u, pb.n_bins, pb.dr) for P in pb.poses]
    ra = assign(fp, times, pb.y, pb.uses_all(), float(pb.theta.L), rm)
    n_all = sum(int((u & (y > 0)).sum()) for u, y in zip(pb.uses_all(), pb.y))
    n_m = sum(int(m.sum()) for m in ra.matched())
    log(f"ratio match: {ra.n_cells} cells in {ra.n_units} visits, {n_m / max(n_all, 1):.0%} of the scored bins "
        f"({n_m} of {n_all}; {rm.cell} m cells, {time.time() - t0:.0f}s)")
    pb = dataclasses.replace(pb, ratio=ra)
    S = render(pb, d)
    gains = ratio_gains(pb, S)
    k_v = visit_shapes(ratio_means(pb, S, gains), ra, rm.model_sd)
    if ra.n_units:
        log(f"ratio match shapes held, model error sd {rm.model_sd}: median {np.median(k_v):.1f}, "
            f"largest {np.max(k_v):.1f}")
    return dataclasses.replace(pb, ratio=dataclasses.replace(ra, shape=k_v), ratio_gain=gains)


def infer(passes: list[Pass], mesh: M.Mesh, prior: Prior, theta: Theta, d0, beta0, u, rounds: int = 3,
          inner: int = 10, calib_every: int = 50, chunk: int = 32, seed: int = 0,
          onset: float = 1.5, laplace_samples: int = 0, trend_search=None, psi_prior=None,
          refl: Reflectivity | None = None, xi0=None, first_return: FirstReturn | None = None,
          horizon: Horizon | None = None, label_floor: bool = True, callback=None,
          min_range: float = 0.0, fit_shape: bool = False, psi_held=None, mode_grid: int = 0,
          shadow_stop: bool = True, temper: float = 1.0, field_steps: int | None = None,
          kappa_std: float | None = None, far_edge: bool = False, far_extent: float = 10.0,
          ratio: RatioMatch | None = None, log=print) -> dict:
    r"""Run the outer loop from (d0, beta0); returns depths, trend, reflectivity, theta and history.

    refl: the prior of the log-reflectivity field :math:`\xi`; with it, :math:`\xi` is an unknown of
    the graph shared by all passes, started at xi0 (default: fitted at d0, in at most `field_steps`
    Gauss--Newton steps, default `inner`) and returned as `xi`. Without it
    each ping-side has its one reflectivity :math:`\kappa_{t,c}` and the passes are not matched.

    trend_search: in the first round, before Gauss--Newton, try these trends and keep the best
    after refinement (`bbf.inference.search.search_trend`): a dict with `slopes_x`, optionally
    `slopes_y` and `steps`. The mean depth kept per candidate is that of the vertices the passes see.

    psi_prior: (mean, sd), each (2, 3), of the sonar parameters :math:`\psi` (beam centre and width
    in radians, linear gain in dB/m, per side); with it, :math:`\psi` is an unknown of the graph,
    started at theta's values. Otherwise it stays at theta's. psi_held: (2, 3) bool, the components
    of a free :math:`\psi` that stay at theta's all the same. The labels and the background are
    relative to the gain, so with the linear gain (the last column) held the scored bins do not
    depend on :math:`\psi`. With every component held, :math:`\psi` is no unknown at all (as without
    psi_prior): on the real data a free beam fitted relief the depths had not yet taken up (a bank
    in one swath moved that side's beam), so it is held until the geometry has moved.

    With `laplace_samples` > 0, also the Laplace posterior standard deviations of the depths
    (`sd_d`), trend (`sd_beta`) and field (`sd_xi`) at the final mode, from that many Monte Carlo samples.

    first_return: the first echoes and the first-return factor's constants
    (`bbf.ping.first_return`); it ties the seabed under the tracks, where the beam predicts little
    energy, to the measured range. Its r0 must come from these passes.

    horizon: the horizon factor's constants (`bbf.ping.horizon`); with it, the labels are scored
    against the line-of-sight margin of the predicted seabed instead of the dark-block shadow
    term, on the blocks `scored_blocks` picks each round from the labels and first echoes.

    label_floor: label the blocks against the noise floor measured from the water column
    (`bbf.ping.shadow.measured_floor`) instead of the model's :math:`g(r) N`, which moves with :math:`\psi`.

    min_range: blocks nearer than this (m) are left out of both the ping and the shadow factor,
    e.g. the sonar's own ringing and sensor-fixed lines in the near water column, which otherwise
    set the spike parameters; it must stay short of the nearest seabed.

    fit_shape: fit the gamma shape :math:`L` and the spike rates :math:`\epsilon_c` of these passes' bins
    (`bbf.inference.calibrate`) instead of keeping theta's, once, in the first round: after the
    field's first fit when that happens here (xi0 not given), since before it the shape takes up
    the reflectivity's contrast as speckle (on the real data :math:`L` fell to 0.45 at 27-bin means,
    against 3 to 5 afterwards). The labels' threshold depends on :math:`L`, so the labels and the
    spike parameters are then taken again, and stay for the later rounds.

    far_edge: find each ping-side's far edge (`bbf.inference.init.far_edge`), where the seabed ends
    for good (on the Örebro passes the shore), label every block beyond it dark, so the ping factor
    leaves it out, and with the horizon factor score the dark blocks up to `far_extent` m beyond it
    (`scored_blocks`). Beyond the edge the intensity is reverberation, still far above the water
    column's floor: against that floor it was labelled lit seabed, which only a dark reflectivity
    or a loose likelihood could explain.

    kappa_std: ties the per-ping gains of a side: the standard deviation of every :math:`\log\kappa_{t,c}`
    about its side's mean (refitted each round, shifted by the known transmit power) is held at this
    value instead of the spread of the fitted modes. The transmit power is known, so what is left to
    a ping is small; a loose prior lets each ping take up along-track shading that is slope.

    mode_grid, shadow_stop, temper: the objective's switches (`Problem`). With `mode_grid` > 0 every
    log-:math:`\kappa` mode is the best found from a grid of starts, so the value does not depend on
    where the last evaluation left the modes (on the real data cold and warm values differed by
    as much as a Gauss--Newton step gains); with `shadow_stop` False the shadow factor's gradient
    is that of its value; `temper` scales the intensity log likelihood.

    ratio: match the passes by the ratio of measured to predicted intensity on shared ground cells
    (`bbf.match.ratio`, `with_ratio`) instead of through a reflectivity field: each round, once theta
    is final for it (after the calibration, the shape fit and `kappa_std`, and after the trend search,
    which scores the plain problem), the scored bins are put in cells and visits at the current depths;
    those of cells seen in at least `RatioMatch.min_visits` visits leave the ping factor for the
    ratio match. The cells, gains and shapes are held through the round. Not together with refl.

    callback(round, d, beta, theta) is called after each round, e.g. to save intermediate results.
    """
    if ratio is not None and refl is not None:
        raise ValueError("the ratio match replaces the reflectivity field: give one of ratio and refl")
    if psi_held is not None and np.all(psi_held):
        psi_prior = psi_held = None
    poses = [p.local_poses(mesh.origin) for p in passes]
    ys = [p.y for p in passes]
    dr = passes[0].dr
    block_dr = theta.block * dr
    d, beta = np.asarray(d0, dtype=float), np.asarray(beta0, dtype=float)
    xi = np.zeros(mesh.n_vertices) if xi0 is None else np.asarray(xi0, dtype=float)
    offsets = [p.log_power() for p in passes]  # transmit power: a known per-ping gain
    history = []
    problem = None
    sub = [np.arange(0, len(p), calib_every) for p in poses]  # the pings theta is fitted on

    def set_up(theta):
        """Labels and first echoes from the data at theta's gain, background and shape: (theta with the measured floor and the water column's lit rate, the problem)."""
        # the echo's onset zone is left out, and the water column before it gives the floor and the
        # lit rate of blocks without a seabed echo, by range
        r0 = [first_echo(p, theta) for p in passes]
        if label_floor:
            theta = dataclasses.replace(theta, floor=measured_floor(ys, r0, theta, dr, onset))
        labels = [labels_of(p, theta) for p in passes]
        far = None
        if far_edge:
            far = [find_far_edge(p, theta, r) for p, r in zip(passes, r0)]
            rb_e = (np.arange(labels[0].shape[-1]) + 0.5) * block_dr
            n_was = sum(int(lab.sum()) for lab in labels)
            labels = [lab & ~(rb_e >= np.nan_to_num(f, nan=np.inf)[..., None]) for lab, f in zip(labels, far)]
            fr_side = [np.mean(np.concatenate([np.isfinite(f[:, c]) for f in far])) for c in (0, 1)]
            log(f"far edge on port {fr_side[0]:.0%}, starboard {fr_side[1]:.0%} of pings; "
                f"{n_was - sum(int(lab.sum()) for lab in labels)} lit blocks beyond it labelled dark")
        valid = [onset_valid(r, lab.shape[-1], block_dr, onset) & beyond(lab.shape[-1], block_dr, min_range)
                 for r, lab in zip(r0, labels)]
        rb = (np.arange(labels[0].shape[-1]) + 0.5) * block_dr
        lit = [np.mean(np.concatenate([lab[:, c][rb[None] > np.asarray(r)[:, c:c + 1]] for lab, r in zip(labels, r0)]))
               for c in (0, 1)]
        log(f"lit past the first echo: port {lit[0]:.0%}, starboard {lit[1]:.0%}"
            f" (labels against the {'measured floor' if label_floor else 'model gain'})")
        theta = dataclasses.replace(theta, wc_lit=water_column_lit_rate(labels, r0, block_dr, onset))
        hz = None
        if horizon is not None:
            hz = dataclasses.replace(horizon, use=scored_blocks(labels, valid, r0, block_dr, horizon.max_gap,
                                                                far=far, far_extent=far_extent))
            log(f"horizon factor scores {sum(int(h.sum()) for h in hz.use)} blocks, "
                f"{sum(int((h & ~lab).sum()) for h, lab in zip(hz.use, labels))} of them dark")
        return theta, Problem(mesh=mesh, prior=prior, theta=theta, poses=poses, y=ys, labels=labels, u=u, dr=dr,
                              chunk=chunk, valid=valid, refl=refl, kappa_offset=offsets, first_return=first_return,
                              horizon=hz, mode_grid=mode_grid, shadow_stop=shadow_stop, temper=temper)

    def refit(theta, base, S, shape=False):
        """theta fitted at the render S on the subset of pings (`calibrate`); with shape, its gamma shape and spike rates too."""
        y_c = np.concatenate([y[s] for y, s in zip(ys, sub)])
        S_c = np.concatenate([Sp[s] for Sp, s in zip(S, sub)])
        use_c = np.concatenate([np.stack([ping_use(lab[t], ys[0].shape[-1], theta.block, valid=val[t]) for t in s])
                                for lab, val, s in zip(base.labels, base.valid, sub)])
        return calibrate(y_c, S_c, use_c, theta, dr, offset=np.concatenate([o[s] for o, s in zip(offsets, sub)]),
                         fit_shape=shape)

    for rnd in range(rounds):
        t0 = time.time()
        # 1. labels (they move with the gain when psi's is free), then theta at the current depths
        theta, base = set_up(theta)
        S = render(base, d, xi=xi)
        theta, fit = refit(theta, base, S)
        if rnd == 0 and refl is not None and xi0 is None:
            fixed = dataclasses.replace(base, theta=theta)
            _, _, _, xi, _ = gauss_newton(fixed, d, beta, jacobians(fixed, d, xi), xi=xi, hold_depths=True,
                                          max_iter=inner if field_steps is None else field_steps, log=log)
            log(f"round {rnd}: reflectivity at the start depths, sd {xi.std():.2f}")
            S = render(base, d, xi=xi)
        if rnd == 0 and fit_shape:
            theta, _ = refit(theta, base, S, shape=True)
            log(f"round {rnd}: shape L {theta.L:.2f}, spike rates {np.asarray(theta.eps).round(3)}")
            theta, base = set_up(theta)  # the labels' threshold follows the shape
            theta, fit = refit(theta, base, S)
        log(f"round {rnd}: theta fit (converged {fit['converged']}): eta {np.asarray(theta.eta)}, "
            f"zeta {np.asarray(theta.zeta)}, ell {np.asarray(theta.ell)}, "
            f"log kappa {np.asarray(theta.log_kappa_mean)} ± {np.asarray(theta.log_kappa_std)}")
        if kappa_std is not None:
            log(f"round {rnd}: gains tied per side, sd {kappa_std} (fitted spread {np.asarray(theta.log_kappa_std).round(2)})")
            theta = dataclasses.replace(theta, log_kappa_std=kappa_std)
        problem = dataclasses.replace(base, theta=theta, psi_prior=psi_prior, psi_held=psi_held)
        # 2. Jacobians, 3. Gauss-Newton
        tj = time.time()
        if rnd == 0 and trend_search is not None:
            from bbf.inference.search import search_trend
            seen = observed_vertices(problem, d)
            fixed = dataclasses.replace(problem, psi_prior=None)  # the search moves the trend only
            d, beta, xi, _ = search_trend(fixed, d, beta, observed=seen, xi=xi, log=log, **trend_search)
            log(f"round {rnd}: trend search picked slopes ({beta[0]:+.3f}, {beta[1]:+.3f})")
        if ratio is not None:
            problem = with_ratio(problem, d, [p.t for p in passes], ratio,
                                 log=lambda s, rnd=rnd: log(f"round {rnd}: {s}"))
        J = jacobians(problem, d, xi)
        Jb = beam_jacobian(problem, d, xi=xi) if problem.free_psi else None
        log(f"round {rnd}: Jacobians in {time.time() - tj:.0f}s")

        def relin(d, psi, xi, pb=problem):
            at = dataclasses.replace(pb, theta=with_psi(pb.theta, psi))
            return jacobians(at, d, xi), (beam_jacobian(pb, d, psi, xi=xi) if pb.free_psi else None)

        d, beta, psi, xi, hist = gauss_newton(problem, d, beta, J, Jb=Jb, xi=xi, max_iter=inner, relin=relin, log=log)
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
