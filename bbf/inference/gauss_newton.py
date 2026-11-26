r"""Inner loop: Gauss--Newton ascent of the log posterior with the labels held fixed.

Each step solves :math:`(H + \lambda D)\,\delta = g` with the augmented GGN matrix :math:`H`
(`bbf.inference.ggn`), factorised by CHOLMOD, where :math:`g` is the exact gradient in
:math:`(d, \xi, \beta)` and zero in :math:`u` (they sit at their modes), and :math:`D` is diagonal. The
weights are recomputed every step, and so are the Jacobians when the caller gives `relin`;
otherwise they stay frozen. With :math:`\psi` free (`Problem.psi_prior`) it is stepped too, with
the beam derivatives of :math:`S` handled like :math:`J`; its components in `Problem.psi_held` stay
where they are. The log-reflectivity field :math:`\xi` (`Problem.refl`) is stepped with the depths,
in the same solve.

Trust region. No vertex may move by more than `D_MAX_STEP` in depth or `XI_MAX_STEP` in :math:`\xi`
in one step, and :math:`\psi` by no more than `PSI_MAX_STEP`. A bin far brighter than predicted,
:math:`y \gg \bar I`, has the gradient :math:`L (y / \bar I - 1)` in :math:`\log \bar I` against the
Fisher curvature :math:`L`, so the Gauss--Newton step overshoots by :math:`e^{\Delta} / \Delta` for a
deficit of :math:`\Delta` log units; and where the beam barely sees the seabed a vertex has a
gradient but next to no curvature, and a step there was seen to move single vertices by 100 m
on the real data. The limits are kept by the damping, not by clipping the step, so the step
stays a damped Gauss--Newton direction whose gain the quadratic model predicts:

* :math:`D` is the diagonal of :math:`H`, raised for every depth and :math:`\xi` to at least :math:`|g_i|`
  over the limit, so that at :math:`\lambda = 1` the gradient alone moves no vertex beyond it, and for
  every depth also to the median diagonal of the vertices the data reach (the field is left its
  own diagonal: with that floor its weakly seen half crawled, and the fit from zero at the start
  depths reached a standard deviation of 0.4 in six steps where 1.4 was to be had);
  :math:`\psi`, whose diagonal sums over every bin and is no scale for it, is damped by :math:`|g_i|`
  over its limit alone;
* :math:`\lambda` is raised until the largest step is inside the limits;
* with :math:`m(\delta) = g^\top \delta - \tfrac{1}{2} \delta^\top (H_{xx} - H_{xu} H_{uu}^{-1} H_{ux}) \delta`
  the gain the model predicts for the step in the variables :math:`x` other than :math:`u` (the
  :math:`u` are re-optimised by every evaluation, and :math:`H_{uu}` is diagonal), a step is accepted
  when the exact objective rises, and :math:`\lambda` follows the ratio :math:`\rho` of the actual to the
  predicted gain: times 4 below 0.25 and after a rejected step, over 3 above 0.75.

:math:`\psi` is also kept within `PSI_LO`..`PSI_HI`: on the real data a single step moved a beam
centre by :math:`26^\circ` and sent a beam width negative.
"""
import time

import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import CholmodNotPositiveDefiniteError, cho_factor

from bbf.inference import ggn
from bbf.inference.objective import Problem, data_term, value, value_and_grad
from bbf.ping.theta import psi_of

XI_MAX_STEP = 2.0  # largest change of a vertex's log reflectivity in one step
D_MAX_STEP = 1.0  # largest change of a vertex's depth (m) in one step
# per side (beam centre, beam width, linear gain): the largest change in one step, and the bounds
PSI_MAX_STEP = np.array([np.radians(3.0), np.radians(5.0), 0.1])
PSI_LO = np.array([np.radians(5.0), np.radians(10.0), -1.0])
PSI_HI = np.array([np.radians(85.0), np.radians(120.0), 2.0])

def predicted_gain(H, g, delta, lay: dict) -> float:
    r""":math:`m(\delta)` of the module docstring for the part of delta outside :math:`u` (its :math:`u` part is ignored)."""
    iu, ip = lay["u"], lay["psi"]
    dx = delta.copy()
    dx[iu:ip] = 0.0
    Hd = H @ dx
    r = Hd[iu:ip]
    return float(g @ dx - 0.5 * (dx @ Hd - r @ (r / H.diagonal()[iu:ip])))


def gauss_newton(problem: Problem, d, beta, J, Jb=None, psi=None, xi=None, hold_depths: bool = False,
                 max_iter: int = 10, tol: float = 1e-5, lam: float = 1e-3, max_tries: int = 10, relin=None,
                 log=print):
    r"""Ascend :math:`\ell` from (d, :math:`\beta`, :math:`\psi`, :math:`\xi`); returns (d, :math:`\beta`, :math:`\psi`, :math:`\xi`, history).

    J: per-ping-side Jacobians J[p][t][c], taken at the start (with the :math:`\xi` columns
    when the problem has the field, and then rescaled with :math:`S` as it moves, see `ggn.assemble`); Jb: beam derivatives of S (`objective.beam_jacobian`), needed when
    :math:`\psi` is free; psi, xi: their starts (default theta's; zero). hold_depths: keep
    (d, :math:`\beta`) where they are and fit the rest, e.g. the reflectivity a seabed implies.
    relin: a function (d, :math:`\psi`, :math:`\xi`) returning (J, Jb) there, called after every accepted
    step; without it J and Jb stay frozen for the whole loop.
    Stops after `max_iter` accepted steps, when an accepted step raises :math:`\ell` by less than
    `tol` times :math:`|\ell|`, or when no damping within `max_tries` increases finds an ascent.
    Each history entry of an accepted step holds the gain the model predicted (`pred`) and the
    ratio of the actual gain to it (`rho`).
    """
    F = data_term(problem)
    Fw = data_term(problem, warm=True)  # log-kappa modes warm-started from the last evaluation
    lay = ggn.layout(problem)
    n, ix, ib, nb, ip = lay["n"], lay["xi"], lay["beta"], lay["n_beta"], lay["psi"]
    free, field = problem.free_psi, problem.refl is not None
    psi = np.asarray(psi_of(problem.theta) if psi is None else psi, dtype=float).copy()
    xi = np.zeros(n) if xi is None else np.asarray(xi, dtype=float).copy()
    d, beta = np.asarray(d, dtype=float).copy(), np.asarray(beta, dtype=float).copy()
    move = np.ones(lay["size"], bool)  # the variables this loop steps
    if hold_depths:
        move[:n] = move[ib:ib + nb] = False
    if free and problem.psi_held is not None:
        move[ip:ip + 6] = ~np.asarray(problem.psi_held, bool).ravel()  # e.g. the gain, held at theta's
    ell, (gd, gb), det = value_and_grad(problem, d, beta, F, psi=psi, xi=xi)
    if not np.isfinite(ell):
        raise FloatingPointError(f"log posterior is {ell} at the start")
    S0 = det["S"] if field else None  # J is taken at the start: with the field its rows follow S (`ggn.assemble`)
    prior_diag = problem.prior.joint_precision().diagonal()[:n]
    psi_limit = np.tile(PSI_MAX_STEP, 2)
    history: list[dict] = [{"ell": ell, "lam": lam, "step": 0.0, "time": 0.0}]
    log(f"  GN start: ell = {ell:.1f}")
    for it in range(max_iter):
        t0 = time.time()
        # the prior's curvature where the depths are: its heavy-tailed slope factor, if any, is reweighted
        H = ggn.assemble(problem, J, det["S"], det["u"], Jb=Jb, psi=psi, S0=S0, fr=det["first_return"],
                         hz=det["horizon"], prior=problem.prior.at(d, beta))
        g = np.zeros(lay["size"])
        g[:n], g[ib:ib + nb] = gd, gb
        if field:
            g[ix:ix + n] = det["grad_xi"]
        if free:
            g[ip:ip + 6] = det["grad_psi"].ravel()
        g[~move] = 0.0
        scale = np.maximum(H.diagonal(), 1e-12)  # the diagonal D of the damping
        reach = scale[:n] > 2.0 * prior_diag  # vertices the data reach
        typical = np.median(scale[:n][reach]) if reach.any() else 0.0
        scale[:n] = np.maximum(np.maximum(scale[:n], typical), np.abs(g[:n]) / D_MAX_STEP)
        if field:
            scale[ix:ix + n] = np.maximum(scale[ix:ix + n], np.abs(g[ix:ix + n]) / XI_MAX_STEP)
        if free:
            scale[ip:ip + 6] = np.abs(g[ip:ip + 6]) / psi_limit
        accepted, new, pred, trial = False, ell, 0.0, {}
        dd, db, dp, dx = np.zeros(n), np.zeros(nb), np.zeros((2, 3)), np.zeros(n)
        tries = 0  # rejected steps; raising lambda into the limits does not count
        for _ in range(6 * max_tries):
            if tries >= max_tries:
                break
            Hd = sp.csc_matrix(H + sp.diags(lam * scale))
            if not move.all():
                Hd = sp.csc_matrix(sp.csr_matrix(Hd[:, move])[move])
            try:
                delta = np.zeros(lay["size"])
                delta[move] = cho_factor(Hd).solve(g[move])
            except CholmodNotPositiveDefiniteError:
                lam *= 10.0  # treat as a rejected step: more damping
                tries += 1
                continue
            big = float(np.max(np.abs(delta[:n]))) / D_MAX_STEP
            if field:
                big = max(big, float(np.max(np.abs(delta[ix:ix + n]))) / XI_MAX_STEP)
            if free:
                big = max(big, float(np.max(np.abs(delta[ip:ip + 6]) / psi_limit)))
            if big > 1.0 and lam < 1e8:
                lam *= 2.0 if big < 4.0 else 4.0
                continue
            dd, db = delta[:n], delta[ib:ib + nb]
            if field:
                dx = delta[ix:ix + n]
            if free:
                dp = np.clip(psi + delta[ip:ip + 6].reshape(2, 3), PSI_LO, PSI_HI) - psi
                delta[ip:ip + 6] = dp.ravel()
            pred = predicted_gain(H, g, delta, lay)
            new = value(problem, d + dd, beta + db, Fw, u0=det["u"], psi=psi + dp, keep=trial, xi=xi + dx)
            if not np.isfinite(new):
                log(f"  GN {it}: non-finite trial value at lambda = {lam:.1e}; damping more")
            elif new > ell:
                accepted = True
                break
            lam *= 4.0
            tries += 1
        if not accepted:
            log(f"  GN {it}: no ascent with lambda up to {lam:.1e}; stopping")
            break
        d, beta, psi, xi = d + dd, beta + db, psi + dp, xi + dx
        gain = new - ell
        rho = gain / pred if pred > 0 else np.nan
        if relin is not None:
            J, Jb = relin(d, psi, xi)
        ell, (gd, gb), det = value_and_grad(problem, d, beta, Fw, u0=det["u"], psi=psi, S=trial["S"], xi=xi)
        if relin is not None and field:
            S0 = det["S"]
        used = lam
        if rho < 0.25:
            lam *= 4.0
        elif rho > 0.75:
            lam = max(lam / 3.0, 1e-8)
        step = float(np.max(np.abs(dd)))
        history.append({"ell": ell, "lam": used, "step": step, "time": time.time() - t0, "psi": psi.copy(),
                        "gain": gain, "pred": pred, "rho": rho, "data": det["data"],
                        "first_return": det["first_return_value"]})
        extra = (f", beam {np.degrees(psi[:, 0]).round(1)}/{np.degrees(psi[:, 1]).round(1)} deg, "
                 f"g1 {psi[:, 2].round(3)} dB/m") if free else ""
        if field:
            extra += f", xi sd {xi.std():.2f}"
        log(f"  GN {it}: ell = {ell:.1f} (+{gain:.1f}, predicted +{pred:.1f}, ratio {rho:.2f}), max |dd| = {step:.3f} m, "
            f"lambda = {used:.1e}{extra}, {time.time() - t0:.0f}s")
        if gain < tol * abs(ell):
            break
    return d, beta, psi, xi, history
