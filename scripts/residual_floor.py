#!/usr/bin/env python3
r"""The water column before the first echo: what the background level says about the gain and the noise.

Usage: python scripts/residual_floor.py PATCHES.npz [--out figs/residual_floor.png]

Before the seabed's first echo :math:`r_0` (`bbf.inference.init.first_echo`) a bin holds no seabed
return, so its level is the model's background :math:`g_c(r) N_c`. The script takes the median over
pings of the raw bins with :math:`r < r_0 - 2` m, per pass, side and transmit power, and prints it
against the model's background, with least-squares fits of :math:`\log \mathrm{median} = a + p \log r`
beyond 10 m (past the transmit ringing).

Two things are read from it. If the level follows the transmit power, the background is
reverberation of the transmission and not receiver noise. And for volume reverberation the
level goes as :math:`g(r)\, r^{-2} e^{-4 \alpha r}`, so its range law gives the receiver gain :math:`g(r)`,
which the seabed echo cannot give on its own (gain, spreading, beam and angular law all vary
across the swath together).

Only the intensities are used; the 3DSS bathymetry is never read.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE

from bbf.inference import init as I
from bbf.io import load_patch
from bbf.ping import Theta, bin_ranges, log_gain

COLORS = {("A", 0): "#2a78d6", ("A", 1): "#eb6834", ("B", 0): "#3a9a5c", ("B", 1): "#52514e"}
SIDES = ("port", "stbd")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figs/residual_floor.png"))
    args = ap.parse_args()
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for name in "AB":
        ps = load_patch(args.patches, name)
        r = np.asarray(bin_ranges(ps.n_bins, ps.dr))
        r0 = I.first_echo(ps, theta)
        power = np.full(ps.n_pings, 100.0) if ps.power is None else ps.power
        for c in (0, 1):
            model = np.asarray(log_gain(r, theta.at(c).gain_db)) + np.log(theta.at(c).noise)
            wc = r[None] < r0[:, c, None] - 2.0
            for pw in np.unique(power):
                sel = wc & (power == pw)[:, None]
                cnt = sel.sum(axis=0)
                ok = cnt >= 100
                med = np.full(ps.n_bins, np.nan)
                med[ok] = np.nanmedian(np.where(sel, ps.y[:, c], np.nan)[:, ok], axis=0)
                far = ok & (r > 10.0)
                line = f"pass {name} {SIDES[c]} at {pw:.0f}% power ({int((power == pw).sum())} pings)"
                if far.sum() > 50:
                    A = np.column_stack([np.ones(far.sum()), np.log(r[far] / 15.0)])
                    co = np.linalg.lstsq(A, np.log(med[far]), rcond=None)[0]
                    line += (f": log median = {co[0]:.2f} + {co[1]:.2f} log(r / 15 m) over {r[far].min():.0f}-{r[far].max():.0f} m; "
                             f"model log(g N) at 15 m = {np.interp(15.0, r, model):.2f}, slope {np.gradient(model, np.log(r))[np.searchsorted(r, 15.0)]:.2f}")
                else:
                    near = ok & (r > 3.0)
                    line += f": log median {np.nanmedian(np.log(med[near])):.2f} over {r[near].min():.0f}-{r[near].max():.0f} m"
                print(line)
                ax[c].plot(r[ok], np.log(med[ok]), color=COLORS[name, c], lw=1.2 if pw >= 50 else 0.6,
                           ls="-" if pw >= 50 else "--", label=f"pass {name}, {pw:.0f}% power")
        for c in (0, 1):
            if name == "B":
                model = np.asarray(log_gain(r, theta.at(c).gain_db)) + np.log(theta.at(c).noise)
                ax[c].plot(r[r > 1], model[r > 1], color="k", lw=1, label=r"model $g(r) N$")
                ref = np.log(r / 15.0) * 1.0 + 19.1
                ax[c].plot(r[r > 8], ref[r > 8], color="k", lw=0.8, ls=":", label=r"$\propto r$: gain $r^3 e^{4\alpha r}$ on volume reverberation")
                ax[c].set(xlabel="range (m)", title=f"Water column before the first echo, {SIDES[c]}", xlim=(0, 24), ylim=(14, 25))
                ax[c].legend(frameon=False, fontsize=8)
                ax[c].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("log of the median intensity")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
