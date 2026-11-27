"""Synthetic twin of the passes: intensities drawn from the model itself at a known seabed and reflectivity.

The unit intensities are rendered at the truth's depths and reflectivity on means of `--level`
fine bins, and each fine bin takes the mean whose centered window holds it (the windows
`coarsen_pass` averages). A coarse level makes a staircase of the echo onset at fine resolution,
which biases the first echoes the detector reads (at level 27 about 0.3 m near); level 3 keeps
the staircase to a few centimeters. Shape, spike rates and per-ping gain spread are the
prototype's, at full resolution.

Usage: python scripts/make_twin.py TRUTH.npz PASSES.npz OUT.npz --beam-response BEAM.npz [--level 3]
"""
import argparse
import dataclasses
import time
from pathlib import Path

import jax
import numpy as np
import tvg_theta as TVG
from infer import load_passes
from prototype_theta import EPS, L_SHAPE
from theta_io import theta_from

from bbf.inference.multiscale import coarsen_pass
from bbf.inference.objective import Problem, render
from bbf.io import load_depths
from bbf.match import Reflectivity
from bbf.ping import Theta, u_grid
from bbf.ping.factor import background, bin_ranges, log_gain

GAIN_SD = 0.2  # per-ping spread of log kappa


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("truth", type=Path, help="a saved fit: its depths and reflectivity are the truth")
    ap.add_argument("passes", type=Path, help="the real passes: poses, transmit power and times")
    ap.add_argument("out", type=Path)
    ap.add_argument("--beam-response", type=Path, required=True)
    ap.add_argument("--level", type=int, default=3, help="fine bins per rendered mean")
    ap.add_argument("--chunk", type=int, default=64, help="pings per render batch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(args.seed)
    K = args.level

    passes = load_passes(args.passes, None)
    mesh, d, ex = load_depths(args.truth)
    xi = ex["xi"]
    br = np.load(args.beam_response)
    th_c = dataclasses.replace(theta_from(ex), beam_knots=br["beam_knots"], beam_log=br["beam_log"])
    cp = [coarsen_pass(p, K) for p in passes]
    pb = Problem(mesh=mesh, prior=None, theta=th_c, poses=[p.local_poses(mesh.origin) for p in cp], y=[p.y for p in cp],
                 labels=[np.zeros((p.n_pings, 2, p.n_bins // th_c.block), bool) for p in cp],
                 u=u_grid(cp[0].n_bins, cp[0].dr, du=max(0.005, cp[0].dr / 3)),
                 dr=cp[0].dr, chunk=args.chunk, refl=Reflectivity(mesh, 2.0, 1.0))
    S = render(pb, d, None, xi)
    log(f"rendered {sum(len(s) for s in S)} pings at the truth on means of {K} bins ({cp[0].dr * 100:.1f} cm)")

    full = Theta(noise=TVG.NOISE, L=L_SHAPE, eps=EPS, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE, ell=2.0,
                 log_kappa_mean=np.asarray(ex["theta_log_kappa_mean"]), log_kappa_std=GAIN_SD, gain_db=TVG.GAIN_DB,
                 noise_slope=TVG.NOISE_SLOPE, noise_power=TVG.NOISE_POWER)
    log(f"simulating with shape {full.L}, spike rates {np.asarray(full.eps)}, "
        f"log kappa mean {np.asarray(full.log_kappa_mean).round(2)}, sd {GAIN_SD}")
    z = np.load(args.passes)
    out = {k: z[k] for k in z.files}
    for i, (p, Sc) in enumerate(zip(passes, S)):
        B = p.n_bins
        idx = np.clip(np.rint(np.arange(B) / K).astype(int), 0, Sc.shape[-1] - 1)
        r = np.asarray(bin_ranges(B, p.dr))
        off = p.log_power()
        y = np.zeros((p.n_pings, 2, B), np.float32)
        for c in (0, 1):
            st = full.at(c)
            g = np.exp(np.asarray(log_gain(r, st.gain_db)))
            u = np.asarray(full.log_kappa_mean)[c] + off + GAIN_SD * rng.standard_normal(p.n_pings)
            bg = np.asarray(background(full, c, r[None, :], off[:, None]))
            I_bar = g[None] * (np.exp(u)[:, None] * np.asarray(Sc)[:, c][:, idx] + bg)
            I_sp = I_bar + g[None] * (st.eta + st.zeta * np.exp(-r / st.ell))[None]
            spike = rng.random(I_bar.shape) < st.eps
            y[:, c] = np.where(spike, rng.gamma(1.0, I_sp), rng.gamma(full.L, I_bar / full.L))
        out[f"{i}/y"] = y
        log(f"pass {i}: real median intensity {np.median(p.y):.3g}, twin {np.median(y):.3g}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
