#!/usr/bin/env python3
r"""Does :math:`\psi` stay physical? One level of the outer loop on a subset of pings, under a choice of model.

Usage: python scripts/residual_psi_gn.py PATCHES.npz START.npz --out runs/psi_<name>.json [options]

From the depths in START.npz (`bbf.io.load_depths`, e.g. the first-echo start of `scripts/infer.py`),
with bins averaged over `--k` and every `--every`-th ping of passes A and B, the script runs
`bbf.inference.loop.infer` as `scripts/infer.py` does for one level: the reflectivity field is fitted
with the depths held, then Gauss--Newton steps the depths, the field and the sonar parameters
:math:`\psi` (beam centre and width, linear gain, per side) from their defaults under weak priors.
It writes :math:`\psi` after every accepted step, and the final state next to it (.npz, `bbf.io.save_depths`).

Model choices: `--theta tvg` takes one gain law for both sides and the water column's background
(`scripts/tvg_theta.py`) and leaves out the near water column; `--shift-b` moves pass B
horizontally; `--beam-response` adds a fitted response across the swath to the Gaussian beam; `--looks` sets the shape of the bin means in place of :math:`kL`; `--g1-sd` is the
prior standard deviation of the linear gain (small: the gain law is held); `--hold-depths` keeps
the depths at the start, so that :math:`\psi` and the field alone answer the data; `--resume` carries
:math:`\psi` and the field over from a state this script wrote, e.g. to go on at a finer level.

Only the intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np
import tvg_theta as TVG
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE

from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import infer, jacobians
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.inference.objective import beam_jacobian
from bbf.io import Pass, load_depths, load_patch, save_depths
from bbf.match import Reflectivity
from bbf.ping import Theta, psi_of, u_grid, with_psi
from bbf.ping.first_return import FirstReturn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

RHO, SIGMA, ALPHA = 5.0, 8.0, 3  # as scripts/infer.py
RHO_XI, SIGMA_XI = 2.0, 1.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def every(ps: Pass, n: int) -> Pass:
    """Every n-th ping of a pass."""
    return dataclasses.replace(ps, poses=ps.poses[::n], y=ps.y[::n], t=ps.t[::n],
                               power=None if ps.power is None else ps.power[::n])


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("start", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=27)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--theta", choices=("prototype", "tvg"), default="prototype")
    ap.add_argument("--shift-b", type=float, nargs=2, default=[0.0, 0.0], metavar=("EAST", "NORTH"))
    ap.add_argument("--looks", type=float, default=None, help="shape of the k-bin means (default k L)")
    ap.add_argument("--g1-sd", type=float, default=0.5, help="prior sd of the linear gain (dB/m)")
    ap.add_argument("--beam", type=float, nargs=2, default=None, metavar=("CENTRE", "WIDTH"),
                    help="start and prior mean of the beam, degrees (default: the 3DSS defaults)")
    ap.add_argument("--hold-depths", action="store_true")
    ap.add_argument("--beam-response", nargs=2, default=None, metavar=("SCORES.json", "MODEL"),
                    help="a free beam response on top of the Gaussian (`Theta.beam_knots`): the piecewise-linear law a "
                         "candidate of scripts/residual_models.py fitted, e.g. beam-spline")
    ap.add_argument("--resume", action="store_true",
                    help="start psi and the field from START's (a state this script wrote); the prior of psi stays at the defaults")
    args = ap.parse_args()

    passes = [every(load_patch(args.patches, n), args.every) for n in "AB"]
    if any(args.shift_b):
        passes[1] = passes[1].shifted(*args.shift_b)
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    min_range = 0.0
    if args.theta == "tvg":
        theta = dataclasses.replace(theta, gain_db=TVG.GAIN_DB, noise=TVG.NOISE, noise_slope=TVG.NOISE_SLOPE,
                                    noise_power=TVG.NOISE_POWER, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE,
                                    log_kappa_mean=TVG.LOG_KAPPA_MEAN)
        min_range = TVG.MIN_RANGE
    if args.beam is not None:
        theta = dataclasses.replace(theta, beta0=np.radians(args.beam[0]), omega=np.radians(args.beam[1]))
    if args.beam_response is not None:
        fitted = next(r for r in json.loads(Path(args.beam_response[0]).read_text()) if r["model"] == args.beam_response[1])
        theta = dataclasses.replace(theta, beam_knots=fitted["knots"]["beam"], beam_log=fitted["params"]["beam"])
    mesh, d, extra = load_depths(args.start)
    beta = np.asarray(extra["beta"], float)
    prior = PR.Prior(mesh, RHO, SIGMA, mu_beta=beta, Sigma_beta=np.array([[100.0]]), alpha=ALPHA)
    first_return = FirstReturn([I.first_echo(p, theta) for p in passes])
    psi_sd = np.array([[np.radians(10.0), np.radians(15.0), args.g1_sd]] * 2)
    psi_prior = (psi_of(theta), psi_sd)
    xi0 = None
    if args.resume:
        theta = with_psi(theta, np.asarray(extra["psi"], float))
        xi0 = np.asarray(extra["xi"], float)
    cp = [coarsen_pass(p, args.k) for p in passes]
    th = coarsen_theta(theta, args.k, looks=args.looks)
    u = u_grid(cp[0].n_bins, cp[0].dr, du=max(0.005, cp[0].dr / 3))
    refl = Reflectivity(mesh, RHO_XI, SIGMA_XI)
    log(f"{sum(p.n_pings for p in cp)} pings, k = {args.k}, theta {args.theta}, shape {th.L:.2f}, shift of B {args.shift_b}, "
        f"g1 sd {args.g1_sd}, depths {'held' if args.hold_depths else 'free'}")
    traj = [psi_of(th).tolist()]
    if args.hold_depths:
        # the problem as the loop sets it up (labels, spike parameters), then psi and the field from zero, depths held
        out = infer(cp, mesh, prior, th, d, beta, u, rounds=1, inner=0, calib_every=20, chunk=32, psi_prior=None,
                    refl=refl, first_return=first_return, min_range=min_range, log=log)
        pb = dataclasses.replace(out["problem"], psi_prior=psi_prior)
        xi = out["xi"]
        psi = psi_of(th)
        hist: list[dict] = []
        for _ in range(args.rounds):
            _, _, psi, xi, hist = gauss_newton(pb, d, beta, jacobians(pb, d, xi), Jb=beam_jacobian(pb, d, psi, xi=xi),
                                               psi=psi, xi=xi, hold_depths=True, max_iter=args.inner, log=log)
            traj += [h["psi"].tolist() for h in hist if "psi" in h]
        ell = hist[-1]["ell"]
        d_out = d
    else:
        out = infer(cp, mesh, prior, th, d, beta, u, rounds=args.rounds, inner=args.inner, calib_every=20, chunk=32,
                    psi_prior=psi_prior, refl=refl, xi0=xi0, first_return=first_return, min_range=min_range, log=log)
        for r in out["history"]:
            traj += [h["psi"].tolist() for h in r["gn"] if "psi" in h]
        ell = out["history"][-1]["gn"][-1]["ell"]
        xi, d_out, beta = out["xi"], out["d"], out["beta"]
    tr = np.array(traj)
    deg = np.degrees(tr[:, :, :2])
    for i in range(len(tr)):
        log(f"  step {i}: beam centre {deg[i, :, 0].round(1)}, width {deg[i, :, 1].round(1)} deg, g1 {tr[i, :, 2].round(3)} dB/m")
    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "psi": traj, "ell": ell,
           "xi_sd": float(np.std(xi)), "depth_change_max": float(np.max(np.abs(d_out - d))),
           "depth_change_rms": float(np.sqrt(np.mean((d_out - d) ** 2)))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    save_depths(args.out.with_suffix(".npz"), mesh, d_out, beta=beta, psi=tr[-1], xi=xi)
    log(f"wrote {args.out}: xi sd {res['xi_sd']:.2f}, depths moved by up to {res['depth_change_max']:.2f} m")


if __name__ == "__main__":
    main()
