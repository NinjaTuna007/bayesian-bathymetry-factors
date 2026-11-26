#!/usr/bin/env python3
r"""Bathymetry from side-scan sonar: inference on patches of the prototype's cache.

Usage: python scripts/infer.py PATCHES.npz [--patches-used A B] [--out runs/<name>.npz] [--hold-out N]
       [--levels K:ROUNDS:STEPS ...] [--temper TAU] [--free-beam] [--no-align] [--bank-start [--bank-top M]]
       [--match {field,ratio} [--cell M] [--model-sd S]]

1. The later passes are moved horizontally onto the first's texture (`bbf.match.align_passes`:
   the navigation of passes minutes apart disagrees by metres, more than the reflectivity field's
   range). Mesh over the passes (h = 0.5 m, 10 m buffer); :math:`z_\mathrm{ref}` = median sonar height.
2. Start: the later first echo of each ping gives the depth under the track; the prior is a
   constant mean (the median depth under the tracks) plus an :math:`\alpha = 3` Matérn field, and
   the start depths are its conditional mean given the points. With `--bank-start`, where the
   other side echoes earlier its seabed rises to that range (`bbf.inference.init.bank_points`);
   on passes A and B of the 2026-07-10 bag that start scored 50 000 nats below the level one. (A planar trend fitted to the first echoes tilts
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
   the seabed under the tracks to the measured range. Every log-kappa mode is searched from a
   grid of starts and the shadow factor is differentiated through it, so the value does not
   depend on the last evaluation and the gradient is the value's.

With `--match ratio` the passes are matched by the ratio match instead (`bbf.match.ratio`): for
ground cells of `--cell` m seen in several visits, only the ratio of measured to predicted intensity
between the visits is scored, with the cell's reflectivity integrated out, so a reflectivity can
explain only what the passes agree on; `--model-sd` is the model error on a visit's log prediction,
which caps each visit's shape. There is no field then, and :math:`\xi = 0`.

Every saved state holds the level's sonar parameters (`scripts/theta_io.py`), the moves given to
the passes and the Gauss--Newton history (gain, predicted gain and their ratio per step), for
`scripts/compare_runs.py`. `--hold-out N` leaves every N-th ping out of the fit, to score.

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
from scipy.spatial import cKDTree
from theta_io import theta_arrays

from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.loop import infer
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import Pass, load_patch, save_depths
from bbf.match import RatioMatch, Reflectivity, align_passes
from bbf.ping import Theta, psi_of, u_grid
from bbf.ping.first_return import FirstReturn
from bbf.ping.horizon import Horizon

jax.config.update("jax_enable_x64", True)

H_MESH, BUFFER = 0.5, 10.0
RHO, SIGMA, ALPHA = 5.0, 8.0, 3
SLOPE = 0.2  # prior sd of each face's depth gradient (`bbf.prior.Prior.slope`)
KAPPA_STD = 0.2  # sd of a ping-side's log gain about its side's mean: the measured ping-to-ping jitter
SIGMA_BETA = np.array([[100.0]])  # the constant mean: sd 10 m about the median first-echo depth
SIGMA_BOWL = np.diag([100.0, 100.0])  # the bowl: the same about the depth under the tracks, sd 10 m on its rise to the rim
LEVELS = [(27, 1, 6), (9, 1, 6), (3, 2, 6)]  # (bins averaged, rounds, Gauss-Newton steps)
RHO_XI, SIGMA_XI = 2.0, 1.0  # range (m) and sd of the log-reflectivity field
PSI_SD = np.array([[np.radians(10.0), np.radians(15.0), 0.5]] * 2)  # beam centre, width (rad), gain (dB/m)
FIELD_STEPS = 18  # Gauss-Newton steps of the field's first fit, at the start depths
MODE_GRID = 49  # starts of each log-kappa mode search
PSI_FREE_BEAM = np.array([[False, False, True]] * 2)  # held with --free-beam: the linear gain, part of the gain law


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_passes(path, names):
    """The passes of a file: patches of the prototype cache by name, or the passes written by legs.py (all of them)."""
    z = np.load(path)
    if "0/poses" not in z.files:
        return [load_patch(path, n) for n in names]
    n = len({k.split("/", 1)[0] for k in z.files})
    return [Pass(name=str(z[f"{i}/name"]), poses=z[f"{i}/poses"], y=z[f"{i}/y"].astype(float), dr=float(z[f"{i}/dr"]),
                 t=z[f"{i}/t"], power=z[f"{i}/power"] if f"{i}/power" in z.files else None) for i in range(n)]


def kept(ps, hold_out: int):
    """The pass without every `hold_out`-th ping (from its ping hold_out // 2); all of it for 0."""
    if hold_out <= 0:
        return ps
    keep = np.arange(ps.n_pings) % hold_out != hold_out // 2
    return dataclasses.replace(ps, poses=ps.poses[keep], y=ps.y[keep], t=ps.t[keep],
                               power=None if ps.power is None else ps.power[keep])


def gn_history(out) -> dict:
    """Per accepted Gauss--Newton step of a level (rounds concatenated): value, gain, predicted gain, ratio, time."""
    steps = [h for r in out["history"] for h in r["gn"] if "rho" in h]
    return {f"gn_{k}": np.array([h[k] for h in steps]) for k in ("ell", "gain", "pred", "rho", "step", "time", "lam")}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--patches-used", nargs="+", default=["A", "B"], dest="names")
    ap.add_argument("--out", type=Path, default=Path("runs/inference.npz"))
    ap.add_argument("--hold-out", type=int, default=0, help="leave every N-th ping out of the fit (0: none)")
    ap.add_argument("--levels", nargs="+", default=None, metavar="K:ROUNDS:STEPS",
                    help="bins averaged, rounds and Gauss-Newton steps per level (default 27:1:6 9:1:6 3:2:6)")
    ap.add_argument("--temper", type=float, default=1.0, help="factor on the intensity log likelihood")
    ap.add_argument("--bank-start", action="store_true",
                    help="raise a bank in the start where a ping's other side echoes earlier (default: the later echo alone)")
    ap.add_argument("--bank-top", type=float, default=None, metavar="M",
                    help="continue each bank of the start up to M metres below the sonar (default: to where it touches the first echo)")
    ap.add_argument("--free-beam", action="store_true", help="fit the beam centre and width per side (default: held)")
    ap.add_argument("--no-align", action="store_true", help="leave the later passes where the navigation has them")
    ap.add_argument("--h", type=float, default=H_MESH, help="mesh spacing (m)")
    ap.add_argument("--alpha", type=int, default=ALPHA, choices=(2, 3), help="SPDE order of the depth prior")
    ap.add_argument("--slope", type=float, default=SLOPE, help="prior sd of each face's depth gradient; 0: no slope factor")
    ap.add_argument("--slope-nu", type=float, help="degrees of freedom of a Student-t slope factor (heavy tails: banks); "
                                                    "default: Gaussian")
    ap.add_argument("--kappa-std", type=float, default=KAPPA_STD, help="sd of a ping-side's log gain about its side's mean; 0: fitted")
    ap.add_argument("--sigma-xi", type=float, default=SIGMA_XI, help="prior sd of the log-reflectivity field")
    ap.add_argument("--beam-response", type=Path, metavar="FILE.npz",
                    help="each side's measured correction to the Gaussian beam in beam angle (scripts/beam_response.py), held")
    ap.add_argument("--match", choices=("field", "ratio"), default="field",
                    help="how the passes are matched: a shared log-reflectivity field, or the ratio of measured to "
                         "predicted intensity on shared ground cells (bbf.match.ratio)")
    ap.add_argument("--cell", type=float, default=1.0, help="ratio match: side of a ground cell (m)")
    ap.add_argument("--model-sd", type=float, default=0.0,
                    help="ratio match: sd of the model error on a visit's log prediction (0: speckle alone)")
    ap.add_argument("--no-field", action="store_true",
                    help="no log-reflectivity field: each ping-side keeps one reflectivity (its gain), the passes are not matched")
    ap.add_argument("--rho-xi", type=float, default=RHO_XI, help="range of the log-reflectivity field (m)")
    ap.add_argument("--hold-shape", type=float, metavar="L",
                    help="gamma shape of the bins at every level, held instead of fitted")
    ap.add_argument("--hold-eps", type=float, nargs=2, metavar=("PORT", "STBD"),
                    help="spike rate of a level's bins per side, held (with --hold-shape); default: the prototype's per bin, coarsened")
    ap.add_argument("--trend", choices=("constant", "bowl"), default="constant",
                    help="mean depth: one constant, or a bowl across the track (bbf.prior.bowl_design) whose rise to the rim is fitted")
    ap.add_argument("--bowl-scale", type=float, default=None, metavar="M",
                    help="distance from the nearest track at which the bowl reaches its rim "
                         "(default: the far edges' median distance with --far-edge, else 30)")
    ap.add_argument("--far-edge", action="store_true",
                    help="label everything beyond each ping-side's far edge (where the seabed ends for good) dark, "
                         "and start with the seabed at the water surface there; "
                         "with --horizon, score it up to --far-extent m beyond")
    ap.add_argument("--no-far-start", action="store_true",
                    help="with --far-edge, keep the far-edge points out of the start (labels and horizon scoring only)")
    ap.add_argument("--far-rise", type=float, default=0.0, metavar="M",
                    help="height of the water surface above the sonar, for the far-edge points of the start (default 0)")
    ap.add_argument("--far-extent", type=float, default=10.0, metavar="M",
                    help="metres beyond the far edge whose dark blocks the horizon factor scores (default 10)")
    ap.add_argument("--horizon", action="store_true",
                    help="score the lit/dark labels by the horizon factor (line of sight to the seabed) instead of the dark-block term")
    ap.add_argument("--cpu", action="store_true", help="run JAX on the CPU (default: the GPU when there is one)")
    args = ap.parse_args()
    if args.cpu:
        jax.config.update("jax_platforms", "cpu")
    chunk = 32 if jax.default_backend() == "cpu" else 256
    log(f"JAX on {jax.default_backend()}, {chunk} pings per batch")
    t_start = time.time()

    levels = LEVELS if args.levels is None else [tuple(int(v) for v in lv.split(":")) for lv in args.levels]
    passes = [kept(p, args.hold_out) for p in load_passes(args.patches, args.names)]
    raw, moves = list(passes), np.zeros((len(passes), 2))
    theta = Theta(noise=TVG.NOISE, L=L_SHAPE, eps=EPS, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE, ell=2.0,
                  log_kappa_mean=TVG.LOG_KAPPA_MEAN, log_kappa_std=2.0, gain_db=TVG.GAIN_DB,
                  noise_slope=TVG.NOISE_SLOPE, noise_power=TVG.NOISE_POWER)
    if args.beam_response is not None:
        br = np.load(args.beam_response)
        theta = dataclasses.replace(theta, beam_knots=br["beam_knots"], beam_log=br["beam_log"])
        log(f"beam response from {args.beam_response}: {len(br['beam_knots'])} knots, range {np.ptp(br['beam_log'], axis=1).round(2)} log units")
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
        moves = np.array([p.poses[0, :2] - q.poses[0, :2] for p, q in zip(passes, raw)])
    mesh = I.mesh_for_passes(passes, h=args.h, buffer=BUFFER)
    log(f"{sum(p.n_pings for p in passes)} pings in {len(passes)} passes; mesh {mesh.n_vertices} vertices, "
        f"z_ref {mesh.z_ref:.2f}")

    design = None  # set below for the bowl trend, once the points off the tracks are known

    def prior_of(beta):
        return PR.Prior(mesh, RHO, SIGMA, mu_beta=beta, Sigma_beta=SIGMA_BETA if design is None else SIGMA_BOWL, alpha=args.alpha,
                        slope=args.slope or None, slope_nu=args.slope_nu, design=design)

    # start: first echoes under the tracks, the GMRF conditional mean elsewhere
    pts = np.concatenate([I.first_echo_points(p, r) for p, r in zip(passes, r0)])
    n_nadir = len(pts)
    if args.bank_start:  # where the sides echo at different ranges, the nearer side's seabed rises: a bank
        pts = np.concatenate([pts] + [I.bank_points(p, r, top=args.bank_top) for p, r in zip(passes, r0)])
    n_bank = len(pts) - n_nadir
    n_far = 0
    if args.far_edge and not args.no_far_start:  # where a side's seabed ends for good, it has reached the water surface
        pts = np.concatenate([pts] + [I.far_edge_points(p, I.far_edge(p, theta, r), rise=args.far_rise)
                                      for p, r in zip(passes, r0)])
        n_far = len(pts) - n_nadir - n_bank
        log(f"far-edge start: {n_far} points at the water surface ({args.far_rise:.1f} m above the sonar)")
    first_return = FirstReturn(r0)
    log(f"first echoes found on {np.mean(np.concatenate([np.isfinite(r).ravel() for r in r0])):.1%} of ping-sides")
    beta = np.array([np.median(mesh.z_ref - pts[:n_nadir, 2])])
    if args.trend == "bowl":  # the mean depth rises away from the tracks to a rim, at most to the water surface
        tracks = np.concatenate([mesh.to_local(p.poses[:, :2]) for p in passes])
        off = pts[n_nadir:]
        s_off = cKDTree(tracks).query(np.asarray(mesh.to_local(off[:, :2])))[0] if len(off) else np.zeros(0)
        scale = args.bowl_scale
        if scale is None:  # the far edges' distance: there the seabed is at the surface
            scale = float(np.median(s_off[-n_far:])) if n_far else 30.0
        design = PR.bowl_design(mesh, tracks, scale)
        top = beta[0] - args.far_rise  # a rise that brings the rim to the water surface
        rise = 0.0
        if len(off):  # least squares of the points' depths on the bowl's column
            c = -np.minimum(s_off, scale) ** 2 / scale**2
            res = (mesh.z_ref - off[:, 2]) - beta[0]
            rise = float(np.sum(c * res) / max(np.sum(c**2), 1e-12))
        beta = np.array([beta[0], float(np.clip(rise, 0.0, top))])
        log(f"bowl trend: depth {beta[0]:.2f} m under the tracks, rising {beta[1]:.2f} m (least squares {rise:.2f}) to the rim "
            f"at {scale:.1f} m, from {len(off)} points off the tracks")
    d = I.depths_from_points(mesh, pts, prior_of(beta))
    log(f"first-echo start: {n_nadir} points below the tracks and {len(pts) - n_nadir} on banks, trend {np.round(beta, 3)}, "
        f"depth {np.percentile(d, 5):.1f}..{np.percentile(d, 95):.1f} m")
    psi_prior = (psi_of(theta), PSI_SD)
    psi_held = PSI_FREE_BEAM if args.free_beam else np.ones((2, 3), bool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_depths(args.out.with_name(args.out.stem + "_start.npz"), mesh, d, beta=beta)

    # coarse to fine
    out, xi, shape, eps = None, None, None, None
    refl = None if args.no_field or args.match != "field" else Reflectivity(mesh, args.rho_xi, args.sigma_xi)
    ratio = RatioMatch(cell=args.cell, model_sd=args.model_sd) if args.match == "ratio" else None
    log(f"match: {args.match}" + (f" ({args.cell} m cells, model error sd {args.model_sd})" if ratio else ""))
    state = {"moves": moves, "hold_out": args.hold_out, "match": args.match, "cell": args.cell, "model_sd": args.model_sd}
    for li, (k, rounds, inner) in enumerate(levels):
        cp = [coarsen_pass(p, k) for p in passes]
        th = coarsen_theta(theta, k, looks=shape)  # the first level starts from independent bins
        if args.hold_shape is not None:
            th = dataclasses.replace(th, L=args.hold_shape)
        if args.hold_eps is not None:
            th = dataclasses.replace(th, eps=np.asarray(args.hold_eps, dtype=float))
        if eps is not None:
            th = dataclasses.replace(th, eps=eps)
        u = u_grid(cp[0].n_bins, cp[0].dr, du=max(0.005, cp[0].dr / 3))
        log(f"level k = {k}: {cp[0].n_bins} bins of {cp[0].dr * 100:.1f} cm, {len(u)} samples per side")
        extra = {}
        if li == len(levels) - 1:
            extra["laplace_samples"] = 100
        out = infer(cp, mesh, prior_of(beta), th, d, beta, u, rounds=rounds, inner=inner, calib_every=50,
                    chunk=chunk, psi_prior=psi_prior, psi_held=psi_held, refl=refl, xi0=xi,
                    first_return=first_return, min_range=TVG.MIN_RANGE, fit_shape=args.hold_shape is None,
                    horizon=Horizon() if args.horizon else None, mode_grid=MODE_GRID, shadow_stop=False, temper=args.temper, field_steps=FIELD_STEPS,
                    kappa_std=args.kappa_std or None, far_edge=args.far_edge, far_extent=args.far_extent, ratio=ratio,
                    log=log, **extra)
        d, beta, xi = out["d"], out["beta"], out["xi"]
        to = out["theta"]
        theta = dataclasses.replace(theta, eta=np.asarray(to.eta) * k, zeta=np.asarray(to.zeta) * k, ell=to.ell,
                                    log_kappa_mean=to.log_kappa_mean, log_kappa_std=to.log_kappa_std,
                                    beta0=to.beta0, omega=to.omega, gain_db=to.gain_db)
        shape, eps = to.L, np.asarray(to.eps)
        psi = out["psi"]
        state.update(k=k, **theta_arrays(to), **gn_history(out))
        save_depths(args.out.with_name(f"{args.out.stem}_k{k}.npz"), mesh, d, beta=beta, psi=psi,
                    **({} if xi is None else {"xi": xi}), **state)
        log(f"level k = {k} done: trend {np.round(beta, 3)}; beam centre {np.degrees(psi[:, 0]).round(1)}, "
            f"width {np.degrees(psi[:, 1]).round(1)} deg; shape {shape:.2f}, spike rates {eps.round(3)}")

    assert out is not None
    save_depths(args.out, mesh, d, beta=beta, sd_d=out["sd_d"], sd_beta=out["sd_beta"], psi=out["psi"], **({} if xi is None else {"xi": xi}),
                **{k: out[k] for k in ("sd_psi", "sd_xi") if k in out}, **state)
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
