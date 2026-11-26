r"""An instrumented Gauss--Newton loop for the diagnostics: `bbf.inference.gauss_newton` with switches.

Every step logs the gain the GGN model predicts for the step that was applied next to the exact gain.
With :math:`u` the log-kappas (at their modes, re-optimised by every evaluation) and :math:`x` the
rest, the prediction for a step :math:`\delta_x` is

.. math::

    m(\delta_x) = g_x^\top \delta_x - \tfrac{1}{2} \delta_x^\top \bigl(H_{xx} - H_{xu} H_{uu}^{-1} H_{ux}\bigr) \delta_x,

with :math:`H_{uu}` diagonal. Switches: the step limits (per vertex clip, or one scale for the whole
step, or none), damping of :math:`\psi`, the damping update (any ascent, or by the ratio of actual to
predicted gain), and Jacobians refreshed every step instead of frozen.
"""
import time

import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import CholmodNotPositiveDefiniteError, cho_factor

from bbf.inference import ggn
from bbf.inference.gauss_newton import (
    D_MAX_STEP,
    PSI_HI,
    PSI_LO,
    PSI_MAX_STEP,
    XI_MAX_STEP,
)
from bbf.inference.loop import jacobians
from bbf.inference.objective import beam_jacobian, data_term, value, value_and_grad


def predicted_gain(H, g, delta, lay) -> float:
    r""":math:`m(\delta_x)` of the module docstring for the non-u part of delta (its u part is ignored)."""
    iu, ip = lay["u"], lay["psi"]
    dx = delta.copy()
    dx[iu:ip] = 0.0
    Hd = H @ dx
    r = Hd[iu:ip]
    huu = H.diagonal()[iu:ip]
    return float(g @ dx - 0.5 * (dx @ Hd - r @ (r / huu)))


def gn_run(problem, d, beta, psi, xi, J=None, Jb=None, max_iter=6, lam=1e-3, clip="vertex", psi_damp=0.0,
           ratio=False, relin=False, hold=(), shorten=True, floor=0.0, max_tries=10, d_max=D_MAX_STEP, log=print, callback=None):
    r"""Gauss--Newton ascent with diagnostics; returns (d, beta, psi, xi, records).

    clip: "vertex" (each vertex's depth step clipped, as `gauss_newton`), "scale" (the whole step
    shortened until the largest depth step is d_max), "trust" (:math:`\lambda` raised until the largest
    depth step is within d_max) or "none". psi_damp: damping of :math:`\psi`,
    :math:`\lambda\,\mathrm{psi\_damp}` times its Schur complement's diagonal (0: undamped, as
    `gauss_newton`). ratio: update :math:`\lambda` from actual / predicted gain (below 0.25: times 4;
    above 0.75: over 3) instead of dividing by 3 after any ascent. relin: refresh J and Jb at every
    accepted step. hold: variable groups kept fixed, of "d", "xi", "psi". shorten: shorten the whole step
    until :math:`\psi`'s part is within `PSI_MAX_STEP`, as `gauss_newton`. floor: damp every depth and
    :math:`\xi` by at least :math:`\lambda` times this quantile of the diagonal over the vertices the data
    reach (0: by its own diagonal alone, as `gauss_newton`), so vertices with a gradient but next to no
    curvature are held by the damping rather than by a cap.
    """
    F, Fw = data_term(problem), data_term(problem, warm=True)
    lay = ggn.layout(problem)
    n, ix, ib, nb, ip = lay["n"], lay["xi"], lay["beta"], lay["n_beta"], lay["psi"]
    free, field = problem.free_psi, problem.refl is not None
    d, beta, psi, xi = (np.asarray(a, float).copy() for a in (d, beta, psi, xi))
    move = np.ones(lay["size"], bool)
    if "d" in hold:
        move[:n] = move[ib:ib + nb] = False
    if "xi" in hold and field:
        move[ix:ix + n] = False
    if "psi" in hold and free:
        move[ip:ip + 6] = False
    if J is None:
        J = jacobians(problem, d, xi)
        Jb = beam_jacobian(problem, d, psi=psi, xi=xi) if free else None
    ell, (gd, gb), det = value_and_grad(problem, d, beta, F, psi=psi, xi=xi)
    S0 = det["S"] if field else None
    pd_d = problem.prior.joint_precision().diagonal()[:n]
    pd_x = problem.refl.Lambda.diagonal() if field else np.zeros(n)
    recs = [{"ell": ell, "data": det["data"], "fr": det["first_return_value"], "psi": psi.copy(), "xi_sd": xi.std()}]
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
        g[~move] = 0.0
        diag = H.diagonal()
        if floor > 0:
            for a in (0, ix) if field else (0,):
                dv = diag[a:a + n]
                base = pd_d if a == 0 else pd_x
                reach = dv > 2 * base
                if reach.any():
                    # at least the typical diagonal of a vertex the data reach, and enough that the
                    # gradient alone moves no vertex beyond the limit at lambda = 1
                    lim = d_max if a == 0 else XI_MAX_STEP
                    diag[a:a + n] = np.maximum(np.maximum(dv, np.quantile(dv[reach], floor)), np.abs(g[a:a + n]) / lim)
        psi_scale = np.zeros(6)
        if free and psi_damp > 0:
            rest = move.copy()
            rest[ip:ip + 6] = False
            Hr = sp.csc_matrix(sp.csr_matrix(H[:, rest])[rest])
            C = np.asarray(sp.csr_matrix(H[:, ip:ip + 6])[rest].todense())
            Hpp = np.asarray(H[ip:ip + 6][:, ip:ip + 6].todense())
            psi_scale = np.diag(Hpp - C.T @ cho_factor(Hr).solve(C))
        accepted, rec, trial = False, {}, {}
        dd, db, dp, dx = np.zeros(n), np.zeros(nb), np.zeros((2, 3)), np.zeros(n)
        new = ell
        for _ in range(max_tries * (3 if clip == "trust" else 1)):
            damp = lam * np.maximum(diag, 1e-12)
            damp[ip:] = lam * psi_damp * psi_scale if free else 0.0
            Hd = sp.csc_matrix(H + sp.diags(damp))
            Hd = sp.csc_matrix(sp.csr_matrix(Hd[:, move])[move])
            try:
                delta = np.zeros(lay["size"])
                delta[move] = cho_factor(Hd).solve(g[move])
            except CholmodNotPositiveDefiniteError:
                lam *= 10.0
                continue
            raw = delta.copy()
            over = 1.0
            if free and shorten:
                over = max(1.0, float(np.max(np.abs(delta[ip:ip + 6].reshape(2, 3)) / PSI_MAX_STEP)))
                delta = delta / over
            scale = 1.0
            if clip == "trust":
                # a trust region in the largest vertex step: more damping until the step fits, so its
                # direction stays the damped Gauss--Newton one (no clip, no rescaling)
                big = max(float(np.max(np.abs(delta[:n])) / d_max),
                          float(np.max(np.abs(delta[ix:ix + n])) / XI_MAX_STEP) if field else 0.0)
                if big > 1.0 and lam < 1e6:
                    lam *= 2.0 if big < 4.0 else 4.0
                    continue
            if clip == "scale":
                scale = max(1.0, float(np.max(np.abs(delta[:n])) / d_max),
                            float(np.max(np.abs(delta[ix:ix + n])) / XI_MAX_STEP) if field else 1.0)
                delta = delta / scale
            if clip == "vertex":
                delta[:n] = np.clip(delta[:n], -d_max, d_max)
                if field:
                    delta[ix:ix + n] = np.clip(delta[ix:ix + n], -XI_MAX_STEP, XI_MAX_STEP)
            dd, db = delta[:n], delta[ib:ib + nb]
            if field:
                dx = delta[ix:ix + n]
            if free:
                dp = np.clip(psi + delta[ip:ip + 6].reshape(2, 3), PSI_LO, PSI_HI) - psi
                delta[ip:ip + 6] = dp.ravel()
            pred = predicted_gain(H, g, delta, lay)
            new = value(problem, d + dd, beta + db, Fw, u0=det["u"], psi=psi + dp, keep=trial, xi=xi + dx)
            rho = (new - ell) / pred if pred > 0 else np.nan
            rec = {"lam": lam, "pred": pred, "pred_raw": predicted_gain(H, g, raw, lay), "actual": new - ell,
                   "rho": rho, "over": over, "scale": scale, "raw_dd_max": float(np.max(np.abs(raw[:n]))),
                   "raw_dd_gt1": int(np.sum(np.abs(raw[:n]) > d_max)),
                   "n_clip": int(np.sum(np.abs(raw[:n]) / over / scale > d_max)),
                   "raw_dpsi": raw[ip:ip + 6].reshape(2, 3).copy() if free else None}
            if np.isfinite(new) and new > ell and not (ratio and rho < 1e-3):
                accepted = True
                break
            log(f"    rejected at lambda {lam:.1e}: actual {new - ell:+.1f}, predicted {pred:+.1f}")
            lam *= 10.0 if not ratio else 4.0
        if not accepted:
            log(f"  GN {it}: no ascent with lambda up to {lam:.1e}; stopping")
            break
        d, beta, psi, xi = d + dd, beta + db, psi + dp, xi + dx
        if relin:
            J = jacobians(problem, d, xi)
            Jb = beam_jacobian(problem, d, psi=psi, xi=xi) if free else None
        ell, (gd, gb), det = value_and_grad(problem, d, beta, Fw, u0=det["u"], psi=psi, S=trial["S"], xi=xi)
        if relin:
            S0 = det["S"] if field else None
        # the same point from cold log-kappa modes: the value moves with where the modes start
        cold = float(F([np.asarray(x) for x in det["S"]], None, np.asarray(psi))[0]) - det["data"]
        rec["cold_minus_warm"] = cold
        if ratio:
            lam = lam * 4.0 if rec["rho"] < 0.25 else (max(lam / 3.0, 1e-8) if rec["rho"] > 0.75 else lam)
        else:
            lam = max(lam / 3.0, 1e-8)
        rec.update({"ell": ell, "data": det["data"], "fr": det["first_return_value"], "psi": psi.copy(),
                    "xi_sd": float(xi.std()), "dd_max": float(np.max(np.abs(dd))), "time": time.time() - t0})
        recs.append(rec)
        log(f"  GN {it}: ell {ell:.1f} actual {rec['actual']:+.1f} pred {rec['pred']:+.1f} (rho {rec['rho']:.2f}; "
            f"raw-step pred {rec['pred_raw']:+.1f}) lam {rec['lam']:.1e} psi-shorten {rec['over']:.1f} "
            f"scale {rec['scale']:.1f} raw max|dd| {rec['raw_dd_max']:.1f} m, {rec['raw_dd_gt1']} raw > cap, "
            f"{rec['n_clip']} clipped; beam {np.degrees(psi[:, 0]).round(1)}/{np.degrees(psi[:, 1]).round(1)} "
            f"g1 {psi[:, 2].round(3)} xi sd {xi.std():.2f} fr {det['first_return_value']:.0f}; same point from cold modes "
            f"{cold:+.0f} {time.time() - t0:.0f}s")
        if callback is not None:
            callback(it, d, beta, psi, xi, rec)
    return d, beta, psi, xi, recs
