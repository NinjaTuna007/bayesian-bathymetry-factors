r"""The log-kappa modes of the ping factor: where `factor.mode` does not return a maximum of :math:`F(u)`.

Usage: run.sh t5_modes.py K STATE

For every ping-side, :math:`F(u)` on a grid around the returned :math:`\hat u`: the grid maximum, how far
:math:`\hat u` is from it, and what the Laplace value loses or gains there. Draws :math:`F(u)` for the worst.
"""
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import FIGS, log, setup

from bbf.inference.objective import data_term, render
from bbf.ping.factor import _objective
from bbf.ping.theta import with_psi

k, state = int(sys.argv[1]), sys.argv[2]
s = setup(k, state)
pb = s.problem
S = render(pb, s.d, s.psi, s.xi)
Sj = [jnp.asarray(x) for x in S]
v_cold, u_cold = data_term(pb)(Sj, None, jnp.asarray(s.psi))
v_warm, u_warm = data_term(pb, warm=True)(Sj, [jnp.asarray(u) for u in u_cold], jnp.asarray(s.psi))
th = with_psi(pb.theta, s.psi)
grid = jnp.linspace(-8.0, 8.0, 641)
fig, axs = plt.subplots(2, 4, figsize=(20, 8))
for p, nm in enumerate("AB"):
    for c, sd in enumerate(("port", "stbd")):
        st = th.at(c)

        def prof(y, SS, use, off, u, c=c):
            Fu = _objective(th, c, pb.dr, y, SS, use, off, pb.temper)[0]
            vals = jax.vmap(Fu)(u + grid)
            return vals, Fu(u), jax.grad(Fu)(u), -jax.grad(jax.grad(Fu))(u)
        uc = u_cold[p][:, c]
        vals, f0, g0, h0 = (np.asarray(a) for a in jax.jit(jax.vmap(prof))(
            jnp.asarray(pb.y[p][:, c]), Sj[p][:, c], jnp.asarray(pb.uses()[p][:, c]),
            jnp.asarray(s.offsets[p]), uc))
        best = np.asarray(grid)[np.argmax(vals, axis=1)]
        loss = vals.max(axis=1) - f0
        n_use = pb.uses()[p][:, c].sum(axis=1)
        # local maxima of F on the grid
        interior = (vals[:, 1:-1] > vals[:, :-2]) & (vals[:, 1:-1] > vals[:, 2:])
        n_modes = interior.sum(axis=1)
        z = (np.asarray(uc) - st.log_kappa_mean - s.offsets[p]) / st.log_kappa_std
        off = np.abs(best) > 0.05
        log(f"{nm} {sd}: u-hat is more than 0.05 from the maximum of F on {off.mean():.1%} of ping-sides (F lower by "
            f"{loss[off].sum():.0f} nats in total, median {np.median(loss[off]) if off.any() else 0:.1f}); F'' > 0 at u-hat on "
            f"{np.mean(h0 < 0):.1%}; more than one local maximum on {np.mean(n_modes > 1):.1%}; "
            f"scored bins per side median {np.median(n_use):.0f} ({np.median(n_use[off]) if off.any() else 0:.0f} where off); "
            f"|u-hat - prior mean| > 2 sd on {np.mean(np.abs(z) > 2):.1%}")
        ax = axs[p, 2 * c]
        for t in np.argsort(-loss)[:6]:
            ax.plot(np.asarray(uc)[t] + np.asarray(grid), vals[t] - vals[t].max(), lw=0.9)
            ax.plot(np.asarray(uc)[t], f0[t] - vals[t].max(), "ko", ms=4)
            ax.plot(np.asarray(u_warm[p][:, c])[t], np.interp(np.asarray(u_warm[p][:, c])[t] - np.asarray(uc)[t],
                                                              np.asarray(grid), vals[t]) - vals[t].max(), "r^", ms=4)
        ax.set_ylim(-400, 20)
        ax.set_title(f"{nm} {sd}: F(u) - max, six worst; dot cold u-hat, triangle warm", fontsize=9)
        ax.set_xlabel("u = log kappa")
        ax = axs[p, 2 * c + 1]
        ax.plot(best, ".", ms=2)
        ax.set_title(f"{nm} {sd}: grid argmax of F minus u-hat, by ping", fontsize=9)
        ax.set_xlabel("ping")
fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / f"modes_k{k}_{state}.png", dpi=110)
log(f"wrote {FIGS / f'modes_k{k}_{state}.png'}")
