r"""Inner loop: Gauss--Newton ascent of the log posterior with the pairs and labels held fixed.

Each step solves :math:`(H + \lambda\,\mathrm{diag}\,H)\,\delta = g` with the augmented GGN matrix
:math:`H` (`bbf.inference.ggn`), factorised by CHOLMOD, where :math:`g` is the exact gradient in
:math:`(d, \beta)` and zero in :math:`u, \mu` (they sit at their modes). A backtracking line search
on the exact objective guards the piecewise-smooth rendering; :math:`\lambda` shrinks after an
accepted step and grows after a rejected one (Levenberg--Marquardt). The Jacobians are frozen
for the whole loop; the weights are recomputed every step.
"""
import time

import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cho_factor

from bbf.inference import ggn
from bbf.inference.objective import Problem, data_term, value, value_and_grad


def gauss_newton(problem: Problem, d, beta, J, max_iter: int = 10, tol: float = 1e-5, lam: float = 1e-3,
                 max_tries: int = 6, log=print):
    r"""Ascend :math:`\ell` from (d, :math:`\beta`); returns (d, :math:`\beta`, history).

    J: frozen per-ping-side Jacobians J[p][t][c]. Stops after `max_iter` accepted steps, when
    an accepted step raises :math:`\ell` by less than `tol` times :math:`|\ell|`, or when no damping
    within `max_tries` increases finds an ascent.
    """
    F = data_term(problem)
    lay = ggn.layout(problem)
    n = lay["n"]
    d, beta = np.asarray(d, dtype=float).copy(), np.asarray(beta, dtype=float).copy()
    ell, (gd, gb), det = value_and_grad(problem, d, beta, F)
    history = [{"ell": ell, "lam": lam, "step": 0.0, "time": 0.0}]
    log(f"  GN start: ell = {ell:.1f}")
    for it in range(max_iter):
        t0 = time.time()
        H = ggn.assemble(problem, J, det["S"], det["u"], det["mu"])
        g = np.zeros(lay["size"])
        g[:n], g[n:n + 3] = gd, gb
        diag = H.diagonal()
        accepted, new = False, ell
        dd, db = np.zeros(n), np.zeros(3)
        for _ in range(max_tries):
            Hd = sp.csc_matrix(H + sp.diags(lam * np.maximum(diag, 1e-12)))
            delta = cho_factor(Hd).solve(g)
            dd, db = delta[:n], delta[n:n + 3]
            new = value(problem, d + dd, beta + db, F)
            if new > ell:
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            log(f"  GN {it}: no ascent with lambda up to {lam:.1e}; stopping")
            break
        d, beta = d + dd, beta + db
        gain = new - ell
        ell, (gd, gb), det = value_and_grad(problem, d, beta, F)
        lam = max(lam / 3.0, 1e-8)
        step = float(np.max(np.abs(dd)))
        history.append({"ell": ell, "lam": lam, "step": step, "time": time.time() - t0})
        log(f"  GN {it}: ell = {ell:.1f} (+{gain:.1f}), max |dd| = {step:.3f} m, lambda = {lam:.1e}, "
            f"{time.time() - t0:.0f}s")
        if gain < tol * abs(ell):
            break
    return d, beta, history
