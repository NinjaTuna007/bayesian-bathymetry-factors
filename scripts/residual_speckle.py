#!/usr/bin/env python3
r"""Speckle statistics of the raw bins: gamma shape, correlation between bins and pings, effective looks.

Usage: python scripts/residual_speckle.py PATCHES.npz [--out figs/residual_speckle.png]

Each raw bin :math:`y_b` is compared with a local mean :math:`m_b` of its own ping-side: the bins
:math:`G < |b' - b| \le G + W` (a guard of :math:`G` bins on each side, so the bin and its
correlated neighbours are left out). Spikes dominate the plain moments, so the shape :math:`L` is read from the lower spread
:math:`q_{50} - q_{10}` of the log of a bin over the median of eight neighbours (`robust_shape`,
calibrated by simulation), which spikes above the median do not reach. For comparison, with
:math:`z = y / m`, speckle of shape :math:`L` on a mean that is flat over the window gives

.. math::

    \operatorname{Var}[\log z] = \psi_1(L) + \psi_1(n_\mathrm{eff} L), \qquad
    n_\mathrm{eff} = \frac{2W}{1 + 2 \sum_{j \ge 1} \rho_j},

with :math:`\rho_j` the correlation of the intensities at lag :math:`j` bins, which is measured from
:math:`z - 1`. Texture of the mean inside the window adds variance, so the shape found on the
seabed is a lower bound that rises as :math:`W` shrinks; the water column has no texture.

The mean of :math:`k` consecutive bins has :math:`L_k` looks (the same lower spread, over block means),
against the :math:`kL` that `bbf.inference.multiscale.coarsen_theta` assumes for independent bins;
with correlated bins :math:`L_k = kL / (1 + 2 \sum_{j<k} (1 - j/k) \rho_j)`.

Regions per ping-side, from the first echo :math:`r_0` (`bbf.inference.init.first_echo`): the water
column from 10 m (beyond the transmit ringing) to :math:`r_0 - 2` m, and the seabed from
:math:`r_0 + 3` m on where the local mean is over 30 times the noise.

Only the intensities are used; the 3DSS bathymetry is never read.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE
from scipy.optimize import brentq
from scipy.special import polygamma
from scipy.stats import gamma as gamma_dist

from bbf.inference import init as I
from bbf.io import load_patch
from bbf.ping import Theta, bin_ranges, log_gain

BLUE, ORANGE, GREEN, INK = "#2a78d6", "#eb6834", "#3a9a5c", "#52514e"
GUARD = 6
LAGS = 16
NEIGH = 4  # neighbours on each side in `neighbour_ratio`


def local_mean(y, W: int, guard: int = GUARD):
    """Mean of the bins at distance guard+1..guard+W on both sides, along the last axis; NaN near the ends."""
    c = np.cumsum(np.concatenate([np.zeros(y.shape[:-1] + (1,)), y], axis=-1), axis=-1)
    B = y.shape[-1]
    b = np.arange(guard + W, B - guard - W)
    left = c[..., b - guard] - c[..., b - guard - W]
    right = c[..., b + guard + W + 1] - c[..., b + guard + 1]
    out = np.full(y.shape, np.nan)
    out[..., b] = (left + right) / (2 * W)
    return out


def shape_from_log_var(v: float, n_eff: float) -> float:
    r"""Solve :math:`\psi_1(L) + \psi_1(n_\mathrm{eff} L) = v` for :math:`L`."""
    root = brentq(lambda L: polygamma(1, L) + polygamma(1, n_eff * L) - v, 1e-3, 1e3)
    assert isinstance(root, float)
    return root


def neighbour_ratio(Y, ok, guard: int):
    r"""(:math:`\log` of each value over the median of its `NEIGH` neighbours beyond `guard` on each side, where all are in `ok`)."""
    n = Y.shape[-1]
    j = np.arange(guard + NEIGH, n - guard - NEIGH)
    off = np.concatenate([np.arange(guard + 1, guard + NEIGH + 1), -np.arange(guard + 1, guard + NEIGH + 1)])
    nb = np.stack([Y[..., j + o] for o in off], axis=-1)
    good = ok[..., j] & np.stack([ok[..., j + o] for o in off], axis=-1).all(axis=-1)
    return np.log(Y[..., j][good] / np.median(nb[good], axis=-1))


def _spread_curve(rng):
    r"""The lower spread :math:`q_{50} - q_{10}` of `neighbour_ratio` for independent :math:`\mathrm{Gam}(L)` values, on a grid of :math:`L`."""
    Ls = np.geomspace(0.2, 200.0, 40)
    sp = []
    for L in Ls:
        x = rng.gamma(L, 1.0, size=(200000, 2 * NEIGH + 1))
        q = np.percentile(np.log(x[:, 0] / np.median(x[:, 1:], axis=1)), [10, 50])
        sp.append(q[1] - q[0])
    return Ls, np.array(sp)


def robust_shape(logz, curve) -> float:
    r"""Gamma shape whose `neighbour_ratio` has the measured lower spread :math:`q_{50} - q_{10}`: blind to spikes, which sit above the median."""
    Ls, sp = curve
    q = np.percentile(logz, [10, 50])
    return float(np.exp(np.interp(-(q[1] - q[0]), -sp, np.log(Ls))))


def autocorr(e, ok, lags: int, axis: int):
    """Correlation of e with itself shifted by 1..lags along `axis`, over the pairs with both in `ok`."""
    out = []
    e = np.where(ok, e, 0.0)
    for j in range(1, lags + 1):
        a = np.take(e, np.arange(0, e.shape[axis] - j), axis=axis)
        b = np.take(e, np.arange(j, e.shape[axis]), axis=axis)
        m = np.take(ok, np.arange(0, e.shape[axis] - j), axis=axis) & np.take(ok, np.arange(j, e.shape[axis]), axis=axis)
        a, b = a[m], b[m]
        out.append(float(np.mean((a - a.mean()) * (b - b.mean())) / np.sqrt(a.var() * b.var())))
    return np.array(out)


def block_means(y, ok, k: int):
    """Means of whole blocks of k bins along the last axis, and whether every bin of the block is in `ok`."""
    n = y.shape[-1] // k
    return (y[..., : n * k].reshape(*y.shape[:-1], n, k).mean(axis=-1),
            ok[..., : n * k].reshape(*y.shape[:-1], n, k).all(axis=-1))


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figs/residual_speckle.png"))
    args = ap.parse_args()
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    passes = [load_patch(args.patches, n) for n in "AB"]
    dr = passes[0].dr
    r = np.asarray(bin_ranges(passes[0].n_bins, dr))
    ks = [3, 9, 27]
    curve = _spread_curve(np.random.default_rng(0))
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for region in ("water column", "seabed"):
        zs, sides = [], []
        for ps in passes:
            r0 = I.first_echo(ps, theta)
            for c in (0, 1):
                y = ps.y[:, c]
                g = np.exp(np.asarray(log_gain(r, theta.at(c).gain_db))) * theta.at(c).noise
                m = local_mean(y, 40)
                if region == "water column":
                    ok = (r[None] > 10.0) & (r[None] < r0[:, c, None] - 2.0)
                else:
                    ok = (r[None] > r0[:, c, None] + 3.0) & (m > 30 * g[None])
                ok &= np.isfinite(m) & (y > 0)
                zs.append((ps, c, y, ok))
                sides.append(f"{ps.name} {'port' if c == 0 else 'stbd'}")
        print(f"== {region} ==")
        col = 0 if region == "water column" else 1
        for (ps, c, y, ok), name in zip(zs, sides):
            line = f"{name}: {int(ok.sum())} bins;"
            z40 = y / local_mean(y, 40)
            # spikes dominate plain moments: correlations from z clipped at its 99% quantile
            e = np.where(ok, np.minimum(z40, np.percentile(z40[ok], 99)) - 1.0, 0.0)
            rho_r = autocorr(e, ok, LAGS, axis=1)
            rho_t = autocorr(e, ok, 6, axis=0)
            tail = 1 + 2 * rho_r[: GUARD].sum()
            v = float(np.var(np.log(z40[ok])))
            L_var = shape_from_log_var(v, 80 / tail)
            L1 = robust_shape(neighbour_ratio(y, ok, GUARD), curve)
            looks = {}
            for k in ks:
                Yk, okk = block_means(y, ok, k)
                looks[k] = robust_shape(neighbour_ratio(Yk, okk, 1), curve)
            pred = {k: k * L1 / (1 + 2 * sum((1 - j / k) * rho_r[j - 1] for j in range(1, min(k, LAGS + 1)))) for k in ks}
            q = gamma_dist.ppf(1 - 1e-3, L1, scale=1 / L1) / gamma_dist.ppf(0.5, L1, scale=1 / L1)
            zmed = neighbour_ratio(y, ok, GUARD)
            line += (f" L of a bin: {L1:.2f} from the lower spread (blind to spikes), {L_var:.2f} from Var[log z];\n"
                     f"    range lag 1..6 rho: {np.round(rho_r[:6], 3)}; ping lag 1..3 rho: {np.round(rho_t[:3], 3)};\n"
                     f"    looks of k-bin means, k = 3/9/27: measured {looks[3]:.2f}/{looks[9]:.2f}/{looks[27]:.2f}, "
                     f"from L and rho {pred[3]:.2f}/{pred[9]:.2f}/{pred[27]:.2f}, model kL {3 * L_SHAPE:.2f}/{9 * L_SHAPE:.2f}/{27 * L_SHAPE:.2f};\n"
                     f"    bins above the 99.9% quantile of Gam(L): {np.mean(zmed > np.log(q)):.4f} (0.0010 without spikes; model eps {theta.at(c).eps})")
            print(line)
            looks[1] = L1
            colr = {"A port": BLUE, "A stbd": ORANGE, "B port": GREEN, "B stbd": INK}[name]
            ax[col, 0].plot(np.arange(1, LAGS + 1), rho_r, "-o", ms=3, color=colr, label=name)
            ax[col, 1].plot([1, *ks], [looks[k] for k in [1, *ks]], "-o", color=colr, label=name)
            zz = np.sort(z40[ok] / np.median(z40[ok]))[:: max(int(ok.sum()) // 4000, 1)]
            p = (np.arange(len(zz)) + 0.5) / len(zz)
            ax[col, 2].plot(gamma_dist.ppf(p, L1) / gamma_dist.ppf(0.5, L1), zz, color=colr, label=f"{name}, L = {L1:.2f}")
        ax[col, 0].axhline(0, color=INK, lw=0.5)
        ax[col, 0].set(xlabel="lag (bins of 1.32 cm)", ylabel="correlation of intensity", title=f"{region}: correlation along range")
        ax[col, 1].plot([1, *ks], [k * L_SHAPE for k in [1, *ks]], "--", color="k", label="model: k L, L = 0.48")
        ax[col, 1].set(xlabel="bins averaged k", ylabel="looks of the k-bin mean", xscale="log", yscale="log",
                       title=f"{region}: effective looks")
        lim = [1e-4, 30]
        ax[col, 2].plot(lim, lim, "--", color="k", lw=0.8)
        ax[col, 2].set(xscale="log", yscale="log", xlabel="quantile of Gam(L) over its median", ylabel="quantile of y / local mean, over its median",
                       title=f"{region}: distribution of a bin about its local mean")
        for a in ax[col]:
            a.legend(frameon=False)
            a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
