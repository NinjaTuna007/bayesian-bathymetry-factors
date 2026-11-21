#!/usr/bin/env python3
r"""Bathymetry from side-scan sonar: inference on patches of the prototype's cache.

Usage: python scripts/infer.py PATCHES.npz [--patches A B] [--out runs/<name>.npz]

1. The later passes are moved horizontally onto the first's texture (`bbf.match.align_passes`:
   the navigation of passes minutes apart disagrees by metres, more than the reflectivity field's
   range). Mesh over the passes (h = 0.5 m, 10 m buffer); :math:`z_\mathrm{ref}` = median sonar height.
2. Start: the first echo of each ping-side gives the depth under the track; the prior is a
   constant mean (their median) plus an :math:`\alpha = 3` Matérn field, and the start depths are
   its conditional mean given the points. (A planar trend fitted to the first echoes tilts
   with the relief under the tracks and extrapolated to depths of :math:`-2` to 40 m.)
   (Planes fitted to the intensities were tried and falsified: passes disagreed by metres, as
   the slopes absorbed errors in the gain and beam pattern.)
3. Coarse to fine in range: k = 27, then k = 9, then k = 3, each
   running the outer loop (refit of the spike parameters and the log-kappa prior,
   Gauss--Newton); Laplace standard deviations at the end. The passes are matched through a
   log-reflectivity field :math:`\xi` shared by all of them (`bbf.match.Reflectivity`), carried
   between levels. The gamma shape and the spike rates are fitted per level, after the field's
   first fit (neighbouring bins are correlated, so the means of k bins do not have the shape
   :math:`kL`), and start each level from the level before. The gain is one fixed law for both sides and the background follows the range
   and the transmit power as the water column does (`scripts/tvg_theta.py`); blocks nearer than
   `MIN_RANGE` are left out. The lit/dark labels are relative to that gain and background, so the
   scored bins do not depend on the sonar parameters :math:`\psi`. The beam is held at the 3DSS
   defaults (centre :math:`50^\circ`, width :math:`60^\circ`): free, it fitted the relief the depths had
   not yet taken up. `--free-beam` makes the beam centre and width per side unknowns of the graph
   with weak priors; the linear gain is part of the gain law and stays held. The
   transmit power enters each ping's log-kappa prior as a known offset. The first echoes, detected
   once at full resolution, enter as the first-return factor (`bbf.ping.first_return`), which ties
   the seabed under the tracks to the measured range.

Uses only the intensities and the navigation; the 3DSS bathymetry is never read.
"""
import argparse
import dataclasses
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import tvg_theta as TVG
from prototype_theta import EPS, L_SHAPE

from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.loop import infer
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import load_patch, save_depths
from bbf.match import Reflectivity, align_passes
from bbf.ping import Theta, psi_of, u_grid
from bbf.ping.first_return import FirstReturn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # float64 is faster on this CPU than on the Turing GPU

H_MESH, BUFFER = 0.5, 10.0
RHO, SIGMA, ALPHA = 5.0, 8.0, 3
SIGMA_BETA = np.array([[100.0]])  # the constant mean: sd 10 m about the median first-echo depth
LEVELS = [(27, 1, 6), (9, 1, 6), (3, 2, 6)]  # (bins averaged, rounds, Gauss-Newton steps)
RHO_XI, SIGMA_XI = 2.0, 1.0  # range (m) and sd of the log-reflectivity field
PSI_SD = np.array([[np.radians(10.0), np.radians(15.0), 0.5]] * 2)  # beam centre, width (rad), gain (dB/m)
PSI_FREE_BEAM = np.array([[False, False, True]] * 2)  # held with --free-beam: the linear gain, part of the gain law


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--patches-used", nargs="+", default=["A", "B"], dest="names")
    ap.add_argument("--out", type=Path, default=Path("runs/inference.npz"))
    ap.add_argument("--free-beam", action="store_true", help="fit the beam centre and width per side (default: held)")
    ap.add_argument("--no-align", action="store_true", help="leave the later passes where the navigation has them")
    args = ap.parse_args()
    t_start = time.time()

    passes = [load_patch(args.patches, n) for n in args.names]
    theta = Theta(noise=TVG.NOISE, L=L_SHAPE, eps=EPS, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE, ell=2.0,
                  log_kappa_mean=TVG.LOG_KAPPA_MEAN, log_kappa_std=2.0, gain_db=TVG.GAIN_DB,
                  noise_slope=TVG.NOISE_SLOPE, noise_power=TVG.NOISE_POWER)
    # first echoes, and with them the horizontal offset of each later pass against the first
    r0 = [I.first_echo(p, theta) for p in passes]
    z_seabed = float(np.median(np.concatenate([I.first_echo_points(p, r) for p, r in zip(passes, r0)])[:, 2]))
    if not args.no_align:
        passes, found = align_passes(passes, r0, z_seabed)
        for p, off in zip(passes[1:], found[1:]):
            if off is None:
                log(f"pass {p.name}: too little shared ground with {passes[0].name} for an offset")
            else:
                log(f"pass {p.name}: texture peak {off.corr:+.2f} ({off.z:.1f} sd; {off.corr_zero:+.2f} unmoved) at "
                    f"({off.move[0]:+.2f}, {off.move[1]:+.2f}) m east, north: {'moved' if off.z >= 5.0 else 'left'}")
    mesh = I.mesh_for_passes(passes, h=H_MESH, buffer=BUFFER)
    log(f"{sum(p.n_pings for p in passes)} pings in {len(passes)} passes; mesh {mesh.n_vertices} vertices, "
        f"z_ref {mesh.z_ref:.2f}")

    def prior_of(beta):
        return PR.Prior(mesh, RHO, SIGMA, mu_beta=beta, Sigma_beta=SIGMA_BETA, alpha=ALPHA)

    # start: first echoes under the tracks, the GMRF conditional mean elsewhere
    pts = np.concatenate([I.first_echo_points(p, r) for p, r in zip(passes, r0)])
    first_return = FirstReturn(r0)
    log(f"first echoes found on {np.mean(np.concatenate([np.isfinite(r).ravel() for r in r0])):.1%} of ping-sides")
    beta = np.array([np.median(mesh.z_ref - pts[:, 2])])
    d = I.depths_from_points(mesh, pts, prior_of(beta))
    log(f"first-echo start: {len(pts)} points, trend {np.round(beta, 3)}, "
        f"depth {np.percentile(d, 5):.1f}..{np.percentile(d, 95):.1f} m")
    psi_prior = (psi_of(theta), PSI_SD)
    psi_held = PSI_FREE_BEAM if args.free_beam else np.ones((2, 3), bool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_depths(args.out.with_name(args.out.stem + "_start.npz"), mesh, d, beta=beta)

    # coarse to fine
    out, xi, shape, eps = None, None, None, None
    refl = Reflectivity(mesh, RHO_XI, SIGMA_XI)
    for li, (k, rounds, inner) in enumerate(LEVELS):
        cp = [coarsen_pass(p, k) for p in passes]
        th = coarsen_theta(theta, k, looks=shape)  # the first level starts from independent bins
        if eps is not None:
            th = dataclasses.replace(th, eps=eps)
        u = u_grid(cp[0].n_bins, cp[0].dr, du=max(0.005, cp[0].dr / 3))
        log(f"level k = {k}: {cp[0].n_bins} bins of {cp[0].dr * 100:.1f} cm, {len(u)} samples per side")
        extra = {}
        if li == len(LEVELS) - 1:
            extra["laplace_samples"] = 100
        out = infer(cp, mesh, prior_of(beta), th, d, beta, u, rounds=rounds, inner=inner, calib_every=50,
                    chunk=32, psi_prior=psi_prior, psi_held=psi_held, refl=refl, xi0=xi,
                    first_return=first_return, min_range=TVG.MIN_RANGE, fit_shape=True, log=log, **extra)
        d, beta, xi = out["d"], out["beta"], out["xi"]
        to = out["theta"]
        theta = dataclasses.replace(theta, eta=np.asarray(to.eta) * k, zeta=np.asarray(to.zeta) * k, ell=to.ell,
                                    log_kappa_mean=to.log_kappa_mean, log_kappa_std=to.log_kappa_std,
                                    beta0=to.beta0, omega=to.omega, gain_db=to.gain_db)
        shape, eps = to.L, np.asarray(to.eps)
        psi = out["psi"]
        save_depths(args.out.with_name(f"{args.out.stem}_k{k}.npz"), mesh, d, beta=beta, psi=psi, xi=xi)
        log(f"level k = {k} done: trend {np.round(beta, 3)}; beam centre {np.degrees(psi[:, 0]).round(1)}, "
            f"width {np.degrees(psi[:, 1]).round(1)} deg; shape {shape:.2f}, spike rates {eps.round(3)}")

    assert out is not None
    save_depths(args.out, mesh, d, beta=beta, sd_d=out["sd_d"], sd_beta=out["sd_beta"], psi=out["psi"], xi=xi,
                **{k: out[k] for k in ("sd_psi", "sd_xi") if k in out})
    log(f"wrote {args.out}; total {(time.time() - t_start) / 60:.0f} min")

    # depth and uncertainty maps over the vertices the passes see
    from bbf.inference.loop import observed_vertices
    seen = observed_vertices(out["problem"], d)
    v = mesh.vertices + np.asarray(mesh.origin)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, val, title, cmap in ((axs[0], d, "depth below z_ref (m)", "viridis_r"),
                                 (axs[1], out["sd_d"], "Laplace standard deviation (m)", "magma")):
        sc = ax.scatter(v[seen, 0], v[seen, 1], c=val[seen], s=1, cmap=cmap)
        fig.colorbar(sc, ax=ax, label=title)
        for p in passes:
            ax.plot(p.poses[:, 0], p.poses[:, 1], "w", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    fig.savefig(Path("figs") / (args.out.stem + "_map.png"), dpi=130)
    log("wrote map")


if __name__ == "__main__":
    main()
