r"""Is the gradient the optimiser uses the derivative of the value it line-searches? Finite differences at a state.

Usage: run.sh t4_fd.py K STATE [GRID [SHADOW_STOP]]

GRID: starts of the log-kappa mode search (`Problem.mode_grid`, default 0); SHADOW_STOP: 0 to differentiate
the shadow factor through log-kappa (`Problem.shadow_stop`, default 1).

1. The value at one point from cold modes, from modes warm-started at the cold ones, and again: the
   log-kappa modes of some ping-sides are not unique or not converged, and the value moves with the start.
2. Central differences of the exact value along each :math:`\psi` coordinate, along the gradient of the
   depths and of :math:`\xi`, against the gradient, with the data term split into the ping and the
   shadow factor.
"""
import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
from common import log, setup

from bbf.inference.objective import data_term, render, value_and_grad
from bbf.ping.factor import _objective, laplace_ping_side
from bbf.ping.first_return import first_return_term
from bbf.ping.shadow import log_shadow_side
from bbf.ping.theta import with_psi

k, state = int(sys.argv[1]), sys.argv[2]
s = setup(k, state)
pb = dataclasses.replace(s.problem, mode_grid=int(sys.argv[3]) if len(sys.argv) > 3 else 0,
                         shadow_stop=(sys.argv[4] != "0") if len(sys.argv) > 4 else True)
log(f"mode grid {pb.mode_grid}, shadow stop {pb.shadow_stop}")
F, Fw = data_term(pb), data_term(pb, warm=True)
ell, (gd, gb), det = value_and_grad(pb, s.d, s.beta, F, psi=s.psi, xi=s.xi)
Sj = [jnp.asarray(x) for x in det["S"]]
psij = jnp.asarray(s.psi)

# 1. the modes ---------------------------------------------------------------------------------------------
v_cold, u_cold = F(Sj, None, psij)
v_warm, u_warm = Fw(Sj, [jnp.asarray(u) for u in u_cold], psij)
v_warm2, u_warm2 = Fw(Sj, [jnp.asarray(u) for u in u_warm], psij)
v_far, u_far = Fw(Sj, [jnp.asarray(u) + 0.5 for u in u_cold], psij)
log(f"data term at one point: cold {float(v_cold):.1f}; warm from the cold modes {float(v_warm) - float(v_cold):+.1f}; "
    f"warm again {float(v_warm2) - float(v_cold):+.1f}; warm from the cold modes + 0.5 {float(v_far) - float(v_cold):+.1f}")
th = with_psi(pb.theta, s.psi)
for p, nm in enumerate("AB"):
    for c, sd in enumerate(("port", "stbd")):
        def dF(y, SS, use, off, u, c=c):
            Fu = _objective(th, c, pb.dr, y, SS, use, off, pb.temper)[0]
            return jax.grad(Fu)(u), -jax.grad(jax.grad(Fu))(u)
        g1, h1 = (np.asarray(a) for a in jax.vmap(dF)(
            jnp.asarray(pb.y[p][:, c]), Sj[p][:, c], jnp.asarray(pb.uses()[p][:, c]), jnp.asarray(s.offsets[p]),
            u_cold[p][:, c]))
        du = np.asarray(u_warm[p][:, c] - u_cold[p][:, c])
        log(f"  {nm} {sd}: cold modes with |F'| > 1: {np.mean(np.abs(g1) > 1):.1%}, with F'' > 0 (not a maximum): "
            f"{np.mean(h1 < 0):.1%}; warm restart moves the mode by > 0.05 on {np.mean(np.abs(du) > 0.05):.1%} "
            f"(max {np.abs(du).max():.2f})")


# 2. finite differences --------------------------------------------------------------------------------------
def parts(d, psi, xi):
    """(total value, ping factor, shadow factor, the rest) from cold modes."""
    S = render(pb, d, psi, xi)
    thp = with_psi(pb.theta, jnp.asarray(psi))
    ping = shadow = 0.0
    for p, Sp in enumerate(S):
        for c in (0, 1):
            def f(y, SS, use, lab, val, off, c=c):
                pg, uh = laplace_ping_side(thp, c, pb.dr, y, SS, use, offset=off, temper=pb.temper, grid=pb.mode_grid)
                return pg, pb.temper * log_shadow_side(thp, c, lab, SS, uh, val)
            a, b = jax.jit(jax.vmap(f))(jnp.asarray(pb.y[p][:, c]), jnp.asarray(Sp[:, c]),
                                        jnp.asarray(pb.uses()[p][:, c]), jnp.asarray(pb.labels[p][:, c]),
                                        jnp.asarray(s.valid[p][:, c]), jnp.asarray(s.offsets[p]))
            ping, shadow = ping + float(a.sum()), shadow + float(b.sum())
    rest = (pb.prior.log_density(d, s.beta) + pb.psi_log_prior(psi)[0] + pb.xi_log_prior(xi)[0]
            + first_return_term(pb.mesh, d, pb.poses, s.first_return)[0])
    return np.array([ping + shadow + rest, ping, shadow, rest])


base = parts(s.d, s.psi, s.xi)
log(f"value {base[0]:.1f} (value_and_grad {ell:.1f}): ping {base[1]:.1f}, shadow {base[2]:.1f}, rest {base[3]:.1f}")
names = ["port centre", "port width", "port g1", "stbd centre", "stbd width", "stbd g1"]
g_psi = det["grad_psi"].ravel()
log("psi: gradient used vs central differences of the value (per rad, rad, dB/m), by factor")
for j, nm in enumerate(names):
    for h in (2e-3, 2e-2):
        e = np.zeros(6)
        e[j] = h
        up, dn = parts(s.d, s.psi + e.reshape(2, 3), s.xi), parts(s.d, s.psi - e.reshape(2, 3), s.xi)
        fd = (up - dn) / (2 * h)
        log(f"  {nm:12s} h {h:.0e}: gradient {g_psi[j]:+12.1f}; differences total {fd[0]:+12.1f} = ping {fd[1]:+12.1f} "
            f"+ shadow {fd[2]:+12.1f} + rest {fd[3]:+9.1f}")
for nm, gvec, which in (("depths", gd, "d"), ("xi", det["grad_xi"], "xi")):
    v = gvec / np.abs(gvec).max()
    for h in (1e-3, 1e-2):
        if which == "d":
            up, dn = parts(s.d + h * v, s.psi, s.xi), parts(s.d - h * v, s.psi, s.xi)
        else:
            up, dn = parts(s.d, s.psi, s.xi + h * v), parts(s.d, s.psi, s.xi - h * v)
        fd = (up - dn) / (2 * h)
        log(f"  along the {nm} gradient (max-normalised), h {h:.0e}: gradient {gvec @ v:+12.1f}; differences total "
            f"{fd[0]:+12.1f} = ping {fd[1]:+12.1f} + shadow {fd[2]:+12.1f} + rest {fd[3]:+9.1f}")
