#!/usr/bin/env python3
r"""Where the residual of the fitted model sits: variance of :math:`\log(y / \bar I)` by covariate.

Usage: python scripts/residual_anova.py TABLE.npz [--out figs/residual_anova.png]

TABLE.npz is written by `scripts/residual_table.py`. Over the bins the ping factor scores, with
:math:`\rho = \log(y / \bar I)` clipped at its 0.5% and 99.5% quantiles, the script prints

* the variance of :math:`\rho` against :math:`\psi_1(kL)`, what the model's speckle allows, and the
  probability integral transform of :math:`y` under the bin mixture;
* for each covariate, binned into groups, the share of the variance its group means remove on bins
  held out of their estimation (a cross-validated :math:`\eta^2`): alone, and added in sequence to
  an additive model of the earlier ones (backfitted);
* the correlation of :math:`\rho` between neighbouring bins and pings.

Groups: side and pass; range, beam angle and incidence angle in steps per side; deciles of
:math:`\hat\xi`, the outboard slope, the echo-to-noise ratio, the altitude, the yaw rate; the
distance to the nearest dark bin; range per side and pass; each ping-side split in thirds of its
range; and square ground cells, shared by the passes or per pass.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from residual_common import SIDES, Bins, cells, load, subset
from scipy.special import ndtri, polygamma

from bbf.io import load_depths
from bbf.mesh import interpolate

COLORS = ["#2a78d6", "#eb6834", "#3a9a5c", "#52514e"]
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
SHRINK = 2.0  # group means are shrunk as n / (n + SHRINK)


def codes(x, edges=None, q: int | None = None) -> np.ndarray:
    """Integer group of each value: by `edges`, or by `q` quantile groups."""
    if q is not None:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, q + 1)[1:-1]))
    return np.searchsorted(np.asarray(edges), x)


def combine(*cs) -> np.ndarray:
    """One code for each distinct combination of the given codes."""
    out = np.zeros(len(cs[0]), np.int64)
    for c in cs:
        out = out * (int(c.max()) + 1) + c
    return np.unique(out, return_inverse=True)[1].ravel()


def factors(b: Bins) -> dict[str, np.ndarray]:
    sp = b.side + 2 * b.pass_id
    snr = b.u_hat + np.log(np.maximum(b.S, 1e-300)) - np.log(np.asarray(b.par["noise"])[b.side])
    rng = codes(b.r, np.arange(0.0, 60.0, 1.0))
    third = np.zeros(b.n, np.int64)
    for ps in np.unique(b.ps):  # thirds of each ping-side's scored range
        m = b.ps == ps
        third[m] = codes(b.r[m], q=3)
    return {
        "side x pass": sp,
        "range | side": combine(rng, b.side),
        "beam angle | side": combine(codes(np.degrees(b.beta), np.arange(-20.0, 120.0, 2.0)), b.side),
        "incidence angle": codes(np.degrees(np.arccos(np.clip(b.cos, 0, 1))), np.arange(0.0, 92.0, 2.0)),
        "range | side x pass": combine(rng, sp),
        "altitude | side": combine(codes(b.alt, q=10), b.side),
        "yaw rate | side": combine(codes(b.yaw_rate, q=10), b.side),
        "transmit power | side": combine(codes(b.offset, q=2), b.side),
        "xi-hat decile": codes(b.xi, q=10),
        "outboard slope decile": codes(b.slope_out, q=10),
        "distance to dark bin": codes(b.shadow_dist, [1.5, 2.5, 3.5, 5.5, 10.5, 20.5]),
        "echo-to-noise decile": codes(snr, q=10),
        "ping-side x range third": combine(b.ps, third),
        "ground cell 2 m, both passes": cells(b.q, 2.0)[0],
        "ground cell 2 m | pass": combine(cells(b.q, 2.0)[0], b.pass_id),
        "ground cell 0.5 m | pass": combine(cells(b.q, 0.5)[0], b.pass_id),
    }


def group_fit(code, resid, train):
    """Shrunk group means of resid over the training bins, as a lookup (unseen groups get 0)."""
    n = int(code.max()) + 1
    cnt = np.bincount(code[train], minlength=n)
    tot = np.bincount(code[train], resid[train], minlength=n)
    return tot / (cnt + SHRINK)


def backfit(fs: list[np.ndarray], rho, train, sweeps: int = 4) -> np.ndarray:
    """Additive fit of rho on the factors (group means from the training bins); returns the fitted values."""
    parts = [np.zeros(len(rho)) for _ in fs]
    for _ in range(sweeps):
        for i, c in enumerate(fs):
            rest = rho - sum(p for j, p in enumerate(parts) if j != i)
            parts[i] = group_fit(c, rest, train)[c]
    return np.sum(parts, axis=0)


def grid(b: Bins, val: np.ndarray) -> np.ndarray:
    """Flat values back on the table's (ping, side, bin) grid; NaN elsewhere."""
    out = np.full(b.par["shape"], np.nan)
    out[b.ping, b.side, b.b] = val
    return out


def lag_corr(a: np.ndarray, axis: int, lags) -> list[float]:
    out = []
    for j in lags:
        x = np.take(a, np.arange(0, a.shape[axis] - j), axis=axis)
        y = np.take(a, np.arange(j, a.shape[axis]), axis=axis)
        m = np.isfinite(x) & np.isfinite(y)
        out.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("table", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figs/residual_anova.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near", type=Path, help="depths file (e.g. the start): keep only bins whose ground is within --tol of it")
    ap.add_argument("--tol", type=float, default=2.0)
    args = ap.parse_args()
    b = load(args.table)
    if args.near is not None:
        mesh, d_ref, _ = load_depths(args.near)
        keep = np.abs(b.depth - np.asarray(interpolate(mesh, d_ref, b.q))) <= args.tol
        print(f"kept {keep.sum()} of {b.n} bins whose ground is within {args.tol} m of {args.near.name}")
        b = subset(b, keep)
    kL = float(b.par["L"])
    rho_raw = b.resid()
    lo, hi = np.percentile(rho_raw, [0.5, 99.5])
    rho = np.clip(rho_raw, lo, hi)
    rho = rho - rho.mean()
    total = rho.var()
    speckle = float(polygamma(1, kL))
    pit = b.pit()
    zs = ndtri(np.clip(pit, 1e-12, 1 - 1e-12))
    print(f"{b.n} scored bins with an echo; k = {int(b.par['k'])}, kL = {kL:.2f}")
    print(f"Var[log(y/I)] = {rho_raw.var():.3f} raw, {total:.3f} clipped at 0.5%/99.5%; median {np.median(rho_raw):+.3f}; "
          f"IQR {np.subtract(*np.percentile(rho_raw, [75, 25])):.3f}")
    print(f"speckle of Gam(kL) alone: Var = psi_1(kL) = {speckle:.3f} = {speckle / total:.1%} of the clipped variance")
    print(f"PIT under the mixture: P(< 0.01) = {np.mean(pit < 0.01):.4f}, P(> 0.99) = {np.mean(pit > 0.99):.4f}, "
          f"P(< 0.1) = {np.mean(pit < 0.1):.3f}, P(> 0.9) = {np.mean(pit > 0.9):.3f}; normal score sd {zs.std():.2f} (1 if calibrated)")
    for c in (0, 1):
        for p in (0, 1):
            m = (b.side == c) & (b.pass_id == p)
            print(f"  pass {'AB'[p]} {SIDES[c]}: {m.sum()} bins, Var {rho[m].var():.3f}, PIT normal score sd {zs[m].std():.2f}")

    rng = np.random.default_rng(args.seed)
    # hold out blocks of 8 bins of a ping-side, so neighbours in range do not leak
    test = rng.random((b.par["shape"][0], 2, b.par["shape"][2] // 8 + 1))[b.ping, b.side, b.b // 8] < 0.2
    train = ~test
    fs = factors(b)
    v_test = rho[test].var()
    print(f"\nshare of Var[rho] removed on held-out bins ({test.sum()} of {b.n}); groups; alone; in sequence (cumulative)")
    alone, seq = {}, {}
    used = []
    for name, c in fs.items():
        fit1 = group_fit(c, rho, train)[c]
        alone[name] = 1 - (rho - fit1)[test].var() / v_test
        used.append(c)
        fit = backfit(used, rho, train)
        seq[name] = 1 - (rho - fit)[test].var() / v_test
        print(f"  {name:32s} {int(c.max()) + 1:7d}  {alone[name]:7.1%}  {seq[name]:7.1%}")
    left = 1 - list(seq.values())[-1]
    print(f"  left after all: {left:.1%} of the variance = {left * total:.3f}; model speckle {speckle:.3f}")

    R = grid(b, rho)
    print("\ncorrelation of rho along range, lag 1/2/4/8/16 bins:", np.round(lag_corr(R, 2, [1, 2, 4, 8, 16]), 3))
    print("correlation of rho across pings, lag 1/2/4/8/16/32:", np.round(lag_corr(R, 0, [1, 2, 4, 8, 16, 32]), 3))

    fig, ax = plt.subplots(2, 4, figsize=(24, 11))
    sp = b.side + 2 * b.pass_id
    views = [(b.r, np.arange(14.0, 52.0, 1.0), "range (m)"),
             (np.degrees(b.beta), np.arange(0.0, 100.0, 2.5), "beam angle below the outboard axis (deg)"),
             (np.degrees(np.arccos(np.clip(b.cos, 0, 1))), np.arange(0.0, 92.0, 2.5), "incidence angle (deg)")]
    for a, (x, edges, label) in zip(ax[0], views):
        for g in range(4):
            m = sp == g
            code = np.searchsorted(edges, x[m])
            cnt = np.bincount(code, minlength=len(edges) + 1)
            mean = np.bincount(code, rho_raw[m], minlength=len(edges) + 1) / np.maximum(cnt, 1)
            okb = cnt[1:-1] > 500
            a.plot((0.5 * (edges[1:] + edges[:-1]))[okb], mean[1:-1][okb], "-o", ms=3, color=COLORS[g],
                   label=f"{'AB'[g // 2]} {SIDES[g % 2]}")
        a.axhline(0, color="k", lw=0.6)
        a.set(xlabel=label, ylabel=r"mean $\log(y / \bar I)$", title=f"Residual by {label.split(' (')[0]}")
        a.legend(frameon=False)
    ax[0, 3].hist(pit, bins=50, density=True, color=COLORS[0])
    ax[0, 3].axhline(1, color="k", lw=0.8)
    ax[0, 3].set(xlabel="probability integral transform under the bin mixture", ylabel="density",
                 title="Calibration of the bin likelihood (flat if calibrated)")
    names = list(seq)
    inc = np.diff([0.0] + [seq[n] for n in names])
    yy = np.arange(len(names))
    ax[1, 0].barh(yy - 0.2, [alone[n] for n in names], height=0.4, color=COLORS[0], label="alone")
    ax[1, 0].barh(yy + 0.2, inc, height=0.4, color=COLORS[1], label="added in sequence")
    ax[1, 0].set_yticks(yy, names)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set(xlabel="share of residual variance removed on held-out bins",
                 title=f"Variance by covariate (left over: {left:.0%}; model speckle {speckle / total:.0%})")
    ax[1, 0].legend(frameon=False)
    ci, centres = cells(b.q, 1.0)
    for a, p in zip(ax[1, 1:3], (0, 1)):
        m = b.pass_id == p
        cnt = np.bincount(ci[m], minlength=len(centres))
        mean = np.bincount(ci[m], rho_raw[m], minlength=len(centres)) / np.maximum(cnt, 1)
        s = cnt >= 10
        sc = a.scatter(centres[s, 0], centres[s, 1], c=mean[s], s=3, marker="s", cmap=DIVERGING, vmin=-1.5, vmax=1.5, linewidths=0)
        fig.colorbar(sc, ax=a, label=r"cell mean of $\log(y / \bar I)$")
        tr = b.par["poses"][b.par["pass_id"] == p]
        a.plot(tr[:, 0], tr[:, 1], color="k", lw=1)
        a.set(xlabel="east (m)", ylabel="north (m)", aspect="equal", title=f"Residual left on the ground, pass {'AB'[p]}")
    lags = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    ax[1, 3].plot(lags, lag_corr(R, 2, lags), "-o", color=COLORS[0], label=f"along range (bins of {float(b.par['dr']) * 100:.0f} cm)")
    ax[1, 3].plot(lags, lag_corr(R, 0, lags), "-o", color=COLORS[1], label="across pings")
    ax[1, 3].axhline(0, color="k", lw=0.6)
    ax[1, 3].set(xlabel="lag", ylabel=r"correlation of $\log(y / \bar I)$", xscale="log",
                 title="Residuals of neighbouring bins (independent in the model)")
    ax[1, 3].legend(frameon=False)
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
