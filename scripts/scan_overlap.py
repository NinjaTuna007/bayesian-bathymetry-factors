#!/usr/bin/env python3
r"""Do the per-ping-side scan planes agree where two passes see the same ground?

Usage: python scripts/scan_overlap.py PATCHES.npz [--patches A B] [--fits runs/scan_fits.npz]

A seabed slope is a property of the ground, so passes looking at it from different directions
must agree on its depth. A slope that stands in for an error in the gain or the beam pattern
belongs to the viewing direction instead and changes sign with it. Each pass's planes are put
onto 1 m cells; where both passes have a cell, their depths are compared, against the outboard
distance at which each pass saw it. A second figure shows the ping residual
:math:`\log y - \log \bar I` at each pass's own planes against the incidence angle
:math:`\varphi`; a trend left after the slope fit points at the angular law.

Uses only the intensities and the navigation; the 3DSS bathymetry is never read.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE

from bbf.inference import init as I
from bbf.inference.loop import labels_of
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.inference.scan import plane_samples, plane_unit_intensity, scan_pass
from bbf.io import load_patch
from bbf.ping import Theta, onset_valid, u_grid
from bbf.ping import geometry as G
from bbf.ping.factor import bin_ranges, mode, predicted
from bbf.ping.shadow import ping_use

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

CELL = 1.0
SIDES = ("port", "stbd")


def plane_cells(ps, fits, stride):
    """Per 1 m cell: (cell keys (m, 2), mean depth below the sonar datum, mean outboard distance)."""
    max_range = ps.n_bins * ps.dr
    rows = []
    for i, t in enumerate(np.arange(0, ps.n_pings, stride)):
        for c in (0, 1):
            D, slope, _ = fits[i, c]
            if not np.isfinite(D):
                continue
            uu = np.arange(0.0, max_range, 0.5)
            P = np.asarray(plane_samples(jnp.asarray(ps.poses[t]), c, D, slope, uu)[0])
            r = np.linalg.norm(P - ps.poses[t, :3], axis=1)
            ok = (r <= max_range) & (P[:, 2] < ps.poses[t, 2])
            rows.append(np.column_stack([P[ok], uu[ok]]))
    a = np.concatenate(rows)
    keys = np.floor(a[:, :2] / CELL).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n = np.bincount(inv)
    return uniq, np.bincount(inv, a[:, 2]) / n, np.bincount(inv, a[:, 3]) / n


def residuals(ps, fits, theta, labels, valid, u, stride):
    r"""(side, :math:`\varphi` in degrees, :math:`\log y - \log \bar I`) of the used bins at each plane."""
    out = []
    rb = np.asarray(bin_ranges(ps.n_bins, ps.dr))
    for i, t in enumerate(np.arange(0, ps.n_pings, stride)):
        use = np.asarray(ping_use(labels[t], ps.n_bins, theta.block, valid=valid[t]))
        for c in (0, 1):
            D, slope, _ = fits[i, c]
            if not np.isfinite(D):
                continue
            pose = jnp.asarray(ps.poses[t])
            S = plane_unit_intensity(pose, c, theta, D, slope, u, ps.n_bins, ps.dr)
            y = jnp.asarray(ps.y[t, c])
            uh, _ = mode(theta, c, ps.dr, y, S, jnp.asarray(use[c]))
            Ib = np.asarray(predicted(S, uh, theta, c, ps.dr)[0])
            P, n = plane_samples(pose, c, D, slope, u)
            s = G.pose_frame(pose)[0]
            e = (P - s) / jnp.linalg.norm(P - s, axis=-1, keepdims=True)
            phi = np.degrees(np.arccos(np.clip(np.asarray(-jnp.sum(n * e, axis=-1)), -1, 1)))
            r = np.asarray(jnp.linalg.norm(P - s, axis=-1))
            order = np.argsort(r)
            phib = np.interp(rb, r[order], phi[order], left=np.nan, right=np.nan)
            k = use[c] & (np.asarray(y) > 0) & np.isfinite(phib) & (np.asarray(S) > 0)
            out.append(np.column_stack([np.full(k.sum(), c), phib[k], np.log(np.asarray(y)[k] / Ib[k])]))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--patches-used", nargs="+", default=["A", "B"], dest="names")
    ap.add_argument("--fits", type=Path, default=Path("runs/scan_fits.npz"))
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    th = coarsen_theta(theta, args.k)
    cp = [coarsen_pass(load_patch(args.patches, n), args.k) for n in args.names]
    labels = [labels_of(p, th) for p in cp]
    r0 = [I.first_echo(p, th) for p in cp]
    valid = [onset_valid(r, lab.shape[-1], th.block * p.dr) for r, lab, p in zip(r0, labels, cp)]
    u = u_grid(cp[0].n_bins, cp[0].dr, du=cp[0].dr / 3)

    if args.fits.exists():
        z = np.load(args.fits)
        fits = [z[n] for n in args.names]
    else:
        fits = [scan_pass(p, th, lab, val, r, u, stride=args.stride, log=print)
                for p, lab, val, r in zip(cp, labels, valid, r0)]
        args.fits.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.fits, **{str(n): f for n, f in zip(args.names, fits)})  # pyright: ignore[reportArgumentType]

    for p, f, r in zip(cp, fits, r0):
        idx = np.arange(0, p.n_pings, args.stride)
        for c in (0, 1):
            sl = f[:, c, 1]
            print(f"{p.name} {SIDES[c]}: slope median {np.nanmedian(sl):+.2f}, |slope| > 0.9 in "
                  f"{np.nanmean(np.abs(sl) > 0.9):.0%}; D - r0 median {np.nanmedian(f[:, c, 0] - r[idx, c]):+.2f} m")

    # depths from the sonar datum: z of the planes, relative to the median sonar height
    z_ref = np.median(np.concatenate([p.poses[:, 2] for p in cp]))
    (ka, za, ua), (kb, zb, ub) = (plane_cells(p, f, args.stride) for p, f in zip(cp, fits))
    da, db = z_ref - za, z_ref - zb
    common, ia, ib = np.intersect1d(ka.view([("", ka.dtype)] * 2), kb.view([("", kb.dtype)] * 2), return_indices=True)
    diff = da[ia] - db[ib]
    print(f"overlap: {len(common)} cells of {CELL:.0f} m; depth A - B median {np.median(diff):+.2f} m, "
          f"MAD {np.median(np.abs(diff - np.median(diff))):.2f} m, |A - B| > 5 m in {np.mean(np.abs(diff) > 5):.0%}")
    near = (ua[ia] < 15) & (ub[ib] < 15)
    if near.any():
        print(f"  both within 15 m outboard: {near.sum()} cells, |A - B| median {np.median(np.abs(diff[near])):.2f} m")

    lo, hi = np.percentile(np.concatenate([da, db]), [2, 98])
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    for ax, k, d, name in ((axs[0, 0], ka, da, args.names[0]), (axs[0, 1], kb, db, args.names[1])):
        sc = ax.scatter(k[:, 0] + 0.5, k[:, 1] + 0.5, c=d, s=1, cmap="viridis_r", vmin=lo, vmax=hi)
        fig.colorbar(sc, ax=ax, label="depth below z_ref (m)")
        for p in cp:
            ax.plot(p.poses[:, 0], p.poses[:, 1], "k", lw=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"scan planes of {name}")
    lim = np.percentile(np.abs(diff), 98)
    ax = axs[1, 0]
    sc = ax.scatter(ka[ia, 0] + 0.5, ka[ia, 1] + 0.5, c=diff, s=1, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(sc, ax=ax, label=f"depth {args.names[0]} - {args.names[1]} (m)")
    for p in cp:
        ax.plot(p.poses[:, 0], p.poses[:, 1], "k", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title("disagreement where both passes have planes")
    ax = axs[1, 1]
    sc = ax.scatter(ua[ia], ub[ib], c=diff, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(sc, ax=ax, label=f"depth {args.names[0]} - {args.names[1]} (m)")
    ax.set_xlabel(f"outboard distance from {args.names[0]} (m)")
    ax.set_ylabel(f"outboard distance from {args.names[1]} (m)")
    ax.set_title("disagreement against where each pass saw the cell")
    fig.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    fig.savefig("figs/scan_overlap.png", dpi=120)
    print("wrote figs/scan_overlap.png")

    # ping residual against incidence angle, at each pass's own planes
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    edges = np.arange(0, 91, 5)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for p, f, lab, val in zip(cp, fits, labels, valid):
        res = residuals(p, f, th, lab, val, u, args.stride)
        for c in (0, 1):
            rc = res[res[:, 0] == c]
            b = np.digitize(rc[:, 1], edges) - 1
            med = [np.median(rc[b == j, 2]) if (b == j).sum() > 50 else np.nan for j in range(len(mid))]
            axs[c].plot(mid, med, "o-", label=p.name)
    for c in (0, 1):
        axs[c].axhline(0, color="k", lw=0.5)
        axs[c].set_xlabel(r"incidence angle $\varphi$ (deg)")
        axs[c].set_title(f"{SIDES[c]}: median log y - log Ī at the scan planes")
        axs[c].legend()
    axs[0].set_ylabel("residual (log intensity)")
    fig.tight_layout()
    fig.savefig("figs/scan_residual_angle.png", dpi=120)
    print("wrote figs/scan_residual_angle.png")


if __name__ == "__main__":
    main()
