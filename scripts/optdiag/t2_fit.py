r"""Short Gauss--Newton fits at one level under variants of the optimiser and the weighting.

Usage: run.sh t2_fit.py NAME K STATE EVERY STEPS [key=value ...]

Keys: temper (factor on the intensity log likelihood), psi (free | fixed), clip (vertex | scale | none),
psi_damp, ratio (0 | 1), relin (0 | 1), grid (starts of the log-kappa modes), shadow_stop (0 | 1), shorten (0 | 1), hold (comma list of d, xi, psi), sigma_xi, psi_sd (factor on the prior sd of psi), lam,
floor (damping floor of the depths, as a quantile of their data diagonal), fr_sigma (sd of the first-return
factor's narrow component, m).
Writes runs/optdiag/fit_NAME.pkl: the records of every step and the final state, with the first-return
residuals by pass and side and the untempered value of every term.
"""
import dataclasses
import pickle
import sys

import numpy as np
from common import OUT, PSI_SD, log, setup
from gn import gn_run

from bbf.inference.objective import data_term, value_and_grad
from bbf.ping.first_return import _batched

name, k, state, every, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
opt = dict(a.split("=") for a in sys.argv[6:])
temper = float(opt.get("temper", 1.0))
sigma_xi = float(opt.get("sigma_xi", 1.0))
tag = "" if temper == 1.0 and sigma_xi == 1.0 else f"_t{temper}_s{sigma_xi}"
s = setup(k, state, every, temper=temper, tag=tag, psi_sd=PSI_SD * float(opt.get("psi_sd", 1.0)), sigma_xi=sigma_xi)
pb = s.problem
pb = dataclasses.replace(pb, mode_grid=int(opt.get("grid", 0)), shadow_stop=opt.get("shadow_stop", "1") == "1")
if opt.get("psi", "free") == "fixed":
    pb = dataclasses.replace(pb, psi_prior=None)
if "fr_sigma" in opt:  # a wider first-return density, whose pull does not die off within a few metres
    pb = dataclasses.replace(pb, first_return=dataclasses.replace(s.first_return, sigma=float(opt["fr_sigma"])))
GRID = 49  # the summaries score every fit alike: untempered, log-kappa modes from a grid of starts


def fr_residuals(d):
    r"""Per pass (n, 2): first echo minus the model's nearest seabed range, :math:`r_0 - \tilde r` (m)."""
    fr = s.first_return
    out = []
    for P, r0 in zip(pb.poses, fr.r0):
        rt = np.stack([np.asarray(_batched(pb.mesh, c, fr.tau)(np.asarray(d), np.asarray(P), np.asarray(fr.u))[0])
                       for c in (0, 1)], axis=1)
        out.append(r0 - rt)
    return out


def summary(d, beta, psi, xi):
    plain = dataclasses.replace(s.problem, temper=1.0, mode_grid=GRID)
    ell, _, det = value_and_grad(plain, d, beta, data_term(plain), psi=psi, xi=xi)
    e = fr_residuals(d)
    out = {"ell_untempered": ell, "data_untempered": det["data"], "fr": det["first_return_value"],
           "prior_d": pb.prior.log_density(d, beta), "prior_xi": pb.xi_log_prior(xi)[0], "fr_res": e}
    log(f"  untempered data {det['data']:.0f}, first return {det['first_return_value']:.0f}, prior d {out['prior_d']:.0f}, "
        f"prior xi {out['prior_xi']:.0f}; d - d0: sd {np.std(d - s.d):.2f} max {np.abs(d - s.d).max():.2f} m")
    for p, nm in enumerate("AB"):
        for c, sd in enumerate(("port", "stbd")):
            x = e[p][:, c]
            x = x[np.isfinite(x)]
            log(f"    first-return residual {nm} {sd}: median {np.median(x):+.2f} m, |res| < 0.3 m on {np.mean(np.abs(x) < 0.3):.0%}")
    return out


log(f"fit {name}: k={k} {state} every={every} {opt}")
start = summary(s.d, s.beta, s.psi, s.xi)
J, Jb = s.jac()
d, beta, psi, xi, recs = gn_run(pb, s.d, s.beta, s.psi, s.xi, J=J, Jb=Jb if pb.free_psi else None, max_iter=steps,
                                lam=float(opt.get("lam", 1e-3)), clip=opt.get("clip", "vertex"),
                                psi_damp=float(opt.get("psi_damp", 0.0)), ratio=opt.get("ratio", "0") == "1",
                                relin=opt.get("relin", "0") == "1", shorten=opt.get("shorten", "1") == "1",
                                hold=tuple(h for h in opt.get("hold", "").split(",") if h),
                                floor=float(opt.get("floor", 0.0)), log=log)
end = summary(d, beta, psi, xi)
(OUT / f"fit_{name}.pkl").write_bytes(pickle.dumps({"opt": opt, "recs": recs, "d": d, "beta": beta, "psi": psi, "xi": xi,
                                                    "d0": s.d, "xi0": s.xi, "start": start, "end": end}))
log("saved")
