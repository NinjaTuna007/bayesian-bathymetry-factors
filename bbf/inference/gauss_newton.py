r"""Inner loop: Gauss--Newton ascent of the log posterior with the labels held fixed.

Each step solves :math:`(H + \lambda\,\mathrm{diag}\,H)\,\delta = g` with the augmented GGN matrix
:math:`H` (`bbf.inference.ggn`), factorised by CHOLMOD, where :math:`g` is the exact gradient in
:math:`(d, \xi, \beta)` and zero in :math:`u` (they sit at their modes). A backtracking line search
on the exact objective guards the piecewise-smooth rendering; :math:`\lambda` shrinks after an
accepted step and grows after a rejected one (Levenberg--Marquardt). The Jacobians are frozen
for the whole loop; the weights are recomputed every step. With :math:`\psi` free
(`Problem.psi_prior`) it is stepped too, with the beam derivatives of :math:`S` frozen like :math:`J`.
The log-reflectivity field :math:`\xi` (`Problem.refl`) is stepped with the depths, in the same solve.

Each step in :math:`\xi` is limited to `XI_MAX_STEP` per vertex. A bin far brighter than predicted,
:math:`y \gg \bar I`, has the gradient :math:`L (y / \bar I - 1)` in :math:`\log \bar I` against the Fisher
curvature :math:`L`, so the Gauss--Newton step overshoots by :math:`e^{\Delta} / \Delta` for a deficit of
:math:`\Delta` log units; and since the spike component bounds what any one bin can lose, the
line search would accept a vertex sent to :math:`\xi \sim 100` by such a step.

Each step in d is likewise limited to `D_MAX_STEP` per vertex. Where the beam barely sees the
seabed the data curvature of a vertex is small against its gradient, and a step there was
seen to move single vertices by 100 m on the real data, accepted because the rest of the step
gained more than the prior charged for the spike.
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

def gauss_newton(problem: Problem, d, beta, J, Jb=None, psi=None, xi=None, hold_depths: bool = False,
                 max_iter: int = 10, tol: float = 1e-5, lam: float = 1e-3, max_tries: int = 10, log=print):
    r"""Ascend :math:`\ell` from (d, :math:`\beta`, :math:`\psi`, :math:`\xi`); returns (d, :math:`\beta`, :math:`\psi`, :math:`\xi`, history).

    J: frozen per-ping-side Jacobians J[p][t][c], taken at the start (with the :math:`\xi` columns
    when the problem has the field, and then rescaled with :math:`S` as it moves, see `ggn.assemble`); Jb: frozen beam derivatives of S (`objective.beam_jacobian`), needed when
    :math:`\psi` is free; psi, xi: their starts (default theta's; zero). hold_depths: keep
    (d, :math:`\beta`) where they are and fit the rest, e.g. the reflectivity a seabed implies.
    Stops after `max_iter` accepted steps, when an accepted step raises :math:`\ell` by less than
    `tol` times :math:`|\ell|`, or when no damping within `max_tries` increases finds an ascent.
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
    ell, (gd, gb), det = value_and_grad(problem, d, beta, F, psi=psi, xi=xi)
    if not np.isfinite(ell):
        raise FloatingPointError(f"log posterior is {ell} at the start")
    S0 = det["S"] if field else None  # J is taken at the start: with the field its rows follow S (`ggn.assemble`)
    history: list[dict] = [{"ell": ell, "lam": lam, "step": 0.0, "time": 0.0}]
    log(f"  GN start: ell = {ell:.1f}")
    for it in range(max_iter):
        t0 = time.time()
        H = ggn.assemble(problem, J, det["S"], det["u"], Jb=Jb, psi=psi, S0=S0, fr=det["first_return"])
        g = np.zeros(lay["size"])
        g[:n], g[ib:ib + nb] = gd, gb
        if field:
            g[ix:ix + n] = det["grad_xi"]
        if free:
            g[ip:ip + 6] = det["grad_psi"].ravel()
        diag = H.diagonal()
        accepted, new, trial = False, ell, {}
        dd, db, dp, dx = np.zeros(n), np.zeros(nb), np.zeros((2, 3)), np.zeros(n)
        for _ in range(max_tries):
            damp = lam * np.maximum(diag, 1e-12)
            # psi is not damped: its diagonal sums over every bin and far exceeds its curvature once
            # the depths and log-kappas are eliminated, so damping it with the rough depths freezes
            # it; its prior keeps the system definite
            damp[ip:] = 0.0
            Hd = sp.csc_matrix(H + sp.diags(damp))
            if hold_depths:
                Hd = sp.csc_matrix(sp.csr_matrix(Hd[:, move])[move])
            try:
                delta = np.zeros(lay["size"])
                delta[move] = cho_factor(Hd).solve(g[move])
            except CholmodNotPositiveDefiniteError:
                lam *= 10.0  # treat as a rejected step: more damping
                continue
            dd, db = np.clip(delta[:n], -D_MAX_STEP, D_MAX_STEP), delta[ib:ib + nb]
            if field:
                dx = np.clip(delta[ix:ix + n], -XI_MAX_STEP, XI_MAX_STEP)
            if free:
                dp = delta[ip:ip + 6].reshape(2, 3)
            new = value(problem, d + dd, beta + db, Fw, u0=det["u"], psi=psi + dp, keep=trial, xi=xi + dx)
            if not np.isfinite(new):
                log(f"  GN {it}: non-finite trial value at lambda = {lam:.1e}; damping more")
            elif new > ell:
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            log(f"  GN {it}: no ascent with lambda up to {lam:.1e}; stopping")
            break
        d, beta, psi, xi = d + dd, beta + db, psi + dp, xi + dx
        gain = new - ell
        ell, (gd, gb), det = value_and_grad(problem, d, beta, Fw, u0=det["u"], psi=psi, S=trial["S"], xi=xi)
        lam = max(lam / 3.0, 1e-8)
        step = float(np.max(np.abs(dd)))
        history.append({"ell": ell, "lam": lam, "step": step, "time": time.time() - t0, "psi": psi.copy()})
        extra = (f", beam {np.degrees(psi[:, 0]).round(1)}/{np.degrees(psi[:, 1]).round(1)} deg, "
                 f"g1 {psi[:, 2].round(3)} dB/m") if free else ""
        if field:
            extra += f", xi sd {xi.std():.2f}"
        log(f"  GN {it}: ell = {ell:.1f} (+{gain:.1f}), max |dd| = {step:.3f} m, lambda = {lam:.1e}{extra}, "
            f"{time.time() - t0:.0f}s")
        if gain < tol * abs(ell):
            break
    return d, beta, psi, xi, history
