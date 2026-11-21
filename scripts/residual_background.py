#!/usr/bin/env python3
r"""The background from the water column: one law for both sides, by range and transmit power.

Usage: python scripts/residual_background.py PATCHES.npz [--out figs/residual_background.png]

Before the first echo a bin holds no seabed, so its mean is the background after gain,
:math:`g_c(r) N_c(r, P)`. Per pass, side and transmit power, the script takes the bins more than
1.5 m short of each ping-side's first echo, averages them over 0.5 m of range (trimmed at the 99%
quantile, which takes fish and spikes out and costs a gamma of shape 0.5 about 7% of its mean),
and fits

.. math::

    \log \overline{y}(r, P) = \log W_{20} + w \log(r / 20) + \gamma \log(P / 100)

on starboard from 4.5 m. Port is compared with that law beyond its sensor-fixed lines (`--port-min`);
where starboard's bank echoes first (pass A), port's water column also holds starboard's leak. With the
gain that cancels spreading and absorption of a seabed echo, :math:`g(r) = (r / 20)^3 e^{4 \alpha (r - 20)}`,
the background before gain is :math:`N (r / 20)^{\nu}` with :math:`\nu = w - 3 - 4 \alpha \cdot 20` to
first order about 20 m, and :math:`N = W_{20}` at full power: the values `scripts/residual_theta.py` uses.
It also prints the level of port's lines over the fitted law by range, the comb that
`bbf.inference.loop.infer`'s `min_range` leaves out.

Only the intensities are used; the 3DSS bathymetry is never read.
"""
import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE

from bbf.inference import init as I
from bbf.io import load_patch
from bbf.ping import Theta

COLORS = ["#2a78d6", "#eb6834", "#3a9a5c", "#52514e", "#8a4fc7", "#c7a11f"]
SIDES = ("port", "stbd")
R_REF, ONSET, STEP = 20.0, 1.5, 0.5
STBD_MIN = 4.5  # starboard is fitted from here (m): nearer, its own transmit ringing adds a few dB


def profile(y, r, r0, edges):
    """Trimmed mean of the water-column bins of y (n, B) per range interval; NaN where fewer than 2000 bins."""
    wc = r[None, :] < (r0[:, None] - ONSET)
    out = np.full(len(edges) - 1, np.nan)
    for i, (a, b) in enumerate(pairwise(edges)):
        v = y[:, (r >= a) & (r < b)][wc[:, (r >= a) & (r < b)]]
        if v.size >= 2000:
            out[i] = v[v <= np.quantile(v, 0.99)].mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figs/residual_background.png"))
    ap.add_argument("--port-min", type=float, default=14.0, help="port's water column is fitted beyond this range (m)")
    ap.add_argument("--alpha", type=float, default=0.0092)
    args = ap.parse_args()
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    edges = np.arange(0.5, 24.01, STEP)
    mid = 0.5 * (edges[1:] + edges[:-1])
    rows, curves = [], []
    for name in "AB":
        ps = load_patch(args.patches, name)
        r = np.arange(ps.n_bins) * ps.dr
        r0 = I.first_echo(ps, theta)
        r0 = np.where(np.isfinite(r0), r0, 0.0)  # no first echo found: no water column taken
        assert ps.power is not None
        for pw in np.unique(ps.power):
            sel = ps.power == pw
            for c in (0, 1):
                prof = profile(ps.y[sel, c], r, r0[sel, c], edges)
                curves.append((f"{name} {SIDES[c]} {pw:.0f}%", prof))
                ok = np.isfinite(prof) & (mid >= (args.port_min if c == 0 else STBD_MIN))
                rows += [(np.log(prof[i]), np.log(mid[i] / R_REF), np.log(pw / 100.0), c) for i in np.flatnonzero(ok)]
    rows = np.array(rows)
    X = np.column_stack([np.ones(len(rows)), rows[:, 1], rows[:, 2]])
    fit = rows[:, 3] == 1  # starboard sets the law; port is checked against it
    coef, *_ = np.linalg.lstsq(X[fit], rows[fit, 0], rcond=None)
    res = rows[:, 0] - X @ coef
    w20, w, gam = float(np.exp(coef[0])), float(coef[1]), float(coef[2])
    nu = w - 3.0 - 4 * args.alpha * R_REF
    print(f"water column of starboard ({fit.sum()} range intervals): W(20 m, 100%) = {w20:.3e}, exponent of range w = {w:.2f}, "
          f"of power gamma = {gam:.2f}; residual sd {res[fit].std():.2f} nats")
    print(f"  port beyond {args.port_min:g} m against it: mean {res[~fit].mean():+.2f} nats, sd {res[~fit].std():.2f} "
          f"over {(~fit).sum()} intervals")
    print(f"with g = (r/20)^3 exp(4 alpha (r - 20)): N = {w20:.3e} at full power, noise_slope nu = {nu:.2f}, noise_power = {gam:.2f}")
    # the prototype's background against it
    for c in (0, 1):
        st = theta.at(c)
        for rr in (10.0, 20.0, 40.0):
            g = np.exp(np.log(10) / 10 * (st.gain_db[0] + st.gain_db[1] * rr + st.gain_db[2] * np.log10(rr)))
            print(f"  prototype g N on {SIDES[c]} at {rr:.0f} m: {10 * np.log10(g * st.noise / (w20 * 0.5**gam * (rr / R_REF)**w)):+.1f} dB "
                  f"against the water column's law at 50% power")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for (label, prof), colr in zip(curves, COLORS):
        ax[0].semilogy(mid, prof, "-", color=colr, label=label)
        pw = float(label.split()[-1][:-1]) / 100.0
        law = w20 * pw**gam * (mid / R_REF) ** w
        ax[1].plot(mid, 10 * np.log10(prof / law), "-", color=colr, label=label)
        if "port" in label:
            over = 10 * np.log10(prof / law)
            comb = mid[np.isfinite(over) & (over > 3.0)]
            if comb.size:
                print(f"  {label}: more than 3 dB over the law from {comb.min():.1f} to {comb.max():.1f} m, at most "
                      f"{np.nanmax(over):+.1f} dB at {mid[np.nanargmax(over)]:.1f} m")
    ax[0].semilogy(mid, w20 * 0.5**gam * (mid / R_REF) ** w, "k--", lw=1, label="fitted law, 50%")
    ax[0].set(xlabel="range (m)", ylabel="trimmed mean intensity of the water column", title="Water column by range")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="range (m)", ylabel="dB over the fitted law", title="Against one law for both sides and powers", ylim=(-6, 30))
    for a in ax:
        a.legend(frameon=False, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
