r"""Is the drift of psi along run fr2 an ascent of one fixed objective? The saved states scored alike.

Usage: run.sh t9_path.py [K]

The run's own log posterior is not comparable between rounds: the labels (which bins the ping factor
scores) and the spike parameters are recomputed with the gain of the moment. Here the level-K problem
is built once, at the start state (labels, spike parameters, log-kappa prior of the start), and every
saved state of the run is scored on it by term, with its own :math:`\psi` and with the prototype's
:math:`\psi` put back, with the log-kappa modes from a grid of starts.
"""
import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
from common import PSI_PROTO, RUNS, log, setup

from bbf.inference.objective import render
from bbf.io import load_depths
from bbf.ping.factor import laplace_ping_side
from bbf.ping.first_return import first_return_term
from bbf.ping.shadow import log_shadow_side
from bbf.ping.theta import with_psi

k = int(sys.argv[1]) if len(sys.argv) > 1 else 27
s = setup(k, "start")
pb = dataclasses.replace(s.problem, mode_grid=49)


def terms(d, beta, psi, xi):
    S = render(pb, d, psi, xi)
    th = with_psi(pb.theta, jnp.asarray(psi))
    out = {}
    for p, nm in enumerate("AB"):
        for c, sd in enumerate(("port", "stbd")):
            def f(y, SS, use, lab, val, off, c=c):
                ping, uh = laplace_ping_side(th, c, pb.dr, y, SS, use, offset=off, grid=pb.mode_grid)
                return ping, log_shadow_side(th, c, lab, SS, uh, val)
            a, b = jax.jit(jax.vmap(f))(jnp.asarray(pb.y[p][:, c]), jnp.asarray(S[p][:, c]),
                                        jnp.asarray(pb.uses()[p][:, c]), jnp.asarray(pb.labels[p][:, c]),
                                        jnp.asarray(s.valid[p][:, c]), jnp.asarray(s.offsets[p]))
            out[f"ping {nm} {sd}"], out[f"shadow {nm} {sd}"] = float(a.sum()), float(b.sum())
    out["first return"] = first_return_term(pb.mesh, d, pb.poses, s.first_return)[0]
    out["prior d"] = pb.prior.log_density(d, beta)
    out["prior xi"] = pb.xi_log_prior(xi)[0]
    out["prior psi"] = pb.psi_log_prior(psi)[0]
    out["total"] = float(sum(out.values()))
    return out


base = terms(s.d, s.beta, s.psi, s.xi)
log("start: " + ", ".join(f"{key} {v:.0f}" for key, v in base.items()))
for state in ("k27", "k9", ""):
    path = RUNS / ("fr2.npz" if state == "" else f"fr2_{state}.npz")
    if not path.exists():
        continue
    _, d, extra = load_depths(path)
    for label, psi in (("own psi", extra["psi"]), ("prototype psi", PSI_PROTO)):
        t = terms(d, extra["beta"], psi, extra["xi"])
        log(f"{state or 'final':5s} {label:13s} (beam {np.degrees(psi[:, 0]).round(0)}/{np.degrees(psi[:, 1]).round(0)}, "
            f"g1 {psi[:, 2].round(2)}): change from the start: "
            + ", ".join(f"{key} {v - base[key]:+.0f}" for key, v in t.items()))
