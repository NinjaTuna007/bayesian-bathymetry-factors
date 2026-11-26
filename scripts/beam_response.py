#!/usr/bin/env python3
r"""Each side's response in beam angle, measured from the intensities at a given seabed.

Usage: python scripts/beam_response.py PASSES.npz STATE.npz [--out runs/<name>.npz] [--fig figs/<name>.png]

With the gain law, the transmit power and the seabed of STATE (e.g. a run's first-echo start) held,
bin :math:`b` of ping :math:`t`, side :math:`c` has

.. math::

    e_b = \log y_b - \log g_c(r_b) - o_t - \log S_b = \log\kappa_{t,c} + \Delta_c(\beta_b) + \xi(q_b) + \text{speckle},

where :math:`S_b` is the rendered unit intensity (with the Gaussian beam of the prototype, Lambert's law,
spreading and absorption), :math:`o_t` the log transmit power, and :math:`\beta_b` the echo-weighted
mean beam angle of the bin. The seabed does not know where the sonar is, so over many pings the
reflectivity and the speckle average out at each :math:`\beta`, and the median of :math:`e_b` by beam angle is
the side's correction :math:`\Delta_c(\beta)` to the Gaussian, up to a constant that the gain
:math:`\kappa` takes. On a near-level seabed the angle of incidence follows the beam angle, so
:math:`\Delta_c` holds the departure from Lambert's law too.

Only bins well above the measured floor, beyond the first echo's onset and `MIN_RANGE`, are used.
The result is written as `Theta.beam_knots` (K,) and `Theta.beam_log` (2, K), for `scripts/infer.py
--beam-response`. Uses only the intensities and the navigation.
"""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tvg_theta as TVG
from infer import load_passes
from prototype_theta import EPS, L_SHAPE

from bbf.inference import init as I
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import load_depths
from bbf.ping import Theta, u_grid
from bbf.ping import geometry as G
from bbf.ping.factor import bin_ranges, log_gain
from bbf.ping.render import curve_length, split_into_bins
from bbf.ping.shadow import measured_floor

jax.config.update("jax_enable_x64", True)
K_BINS, ONSET, ABOVE_FLOOR, MIN_COUNT = 27, 1.5, 4.0, 2000
KNOTS_DEG = np.arange(7.5, 90.1, 2.5)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def side_bins(mesh, d, theta, u, n_bins, dr, side):
    """Jitted (poses (n, 6)) -> unit intensity S (n, n_bins) and the bins' echo-weighted beam angle (n, n_bins)."""
    def one(pose):
        P, n, inside = G.samples(mesh, d, pose, side, u)
        a, r = G.echo(theta, pose, side, P, n)
        s, R = G.pose_frame(pose)
        V = G.horizon_visibility(P, s, u, theta.shadow_margin, theta.wavelength, inside)
        w = jnp.where(inside, a * V * curve_length(P[:, :2]), 0.0) / dr
        axes = R @ jnp.asarray(G.R_MOUNT[side])
        e = (P - s) / r[:, None]
        beta = jnp.arctan2(e @ axes[:, 2], e @ axes[:, 1])
        S = split_into_bins(r, w, n_bins, dr)
        Sb = split_into_bins(r, w * beta, n_bins, dr)
        return S, jnp.where(S > 0, Sb / jnp.where(S > 0, S, 1.0), jnp.nan)
    return jax.jit(lambda poses: jax.lax.map(one, poses, batch_size=256))


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("passes", type=Path)
    ap.add_argument("state", type=Path, help="depths to render at (save_depths file)")
    ap.add_argument("--out", type=Path, default=Path("runs/beam_response.npz"))
    ap.add_argument("--fig", type=Path, default=Path("figs/beam_response.png"))
    args = ap.parse_args()
    passes = load_passes(args.passes, None)
    mesh, d, _ = load_depths(args.state)
    theta = Theta(noise=TVG.NOISE, L=L_SHAPE, eps=EPS, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE, ell=2.0,
                  log_kappa_mean=TVG.LOG_KAPPA_MEAN, log_kappa_std=2.0, gain_db=TVG.GAIN_DB,
                  noise_slope=TVG.NOISE_SLOPE, noise_power=TVG.NOISE_POWER)
    cp = [coarsen_pass(p, K_BINS) for p in passes]
    th = coarsen_theta(theta, K_BINS)
    n_bins, dr = cp[0].n_bins, cp[0].dr
    u = jnp.asarray(u_grid(n_bins, dr, du=max(0.005, dr / 3)))
    r = np.asarray(bin_ranges(n_bins, dr))
    r0 = [I.first_echo(p, th) for p in cp]
    floor = measured_floor([p.y for p in cp], r0, th, dr, ONSET)  # (2, n_bins): one block per bin at this level
    log(f"{sum(p.n_pings for p in cp)} pings in {len(cp)} passes, {n_bins} bins of {dr * 100:.1f} cm")
    edges = np.radians(np.concatenate([KNOTS_DEG - 1.25, KNOTS_DEG[-1:] + 1.25]))
    per_pass = np.full((len(cp), 2, len(KNOTS_DEG)), np.nan)
    pooled: list[list] = [[], []]
    dj = jnp.asarray(d)
    for side in (0, 1):
        f = side_bins(mesh, dj, th, u, n_bins, dr, side)
        lg = np.asarray(log_gain(r, np.asarray(th.gain_db)[side]))
        for i, p in enumerate(cp):
            S, beta = (np.asarray(x) for x in f(jnp.asarray(p.local_poses(mesh.origin))))
            y = p.y[:, side]
            use = ((S > 0) & (y > ABOVE_FLOOR * floor[side][None]) & (r[None] >= TVG.MIN_RANGE)
                   & (r[None] > np.nan_to_num(r0[i][:, side], nan=np.inf)[:, None] + ONSET))
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.log(y) - lg[None] - p.log_power()[:, None] - np.log(S)
            b, ev = beta[use], e[use]
            j = np.digitize(b, edges) - 1
            ok = (j >= 0) & (j < len(KNOTS_DEG))
            for k in range(len(KNOTS_DEG)):
                m = ok & (j == k)
                if m.sum() >= MIN_COUNT // 4:
                    per_pass[i, side, k] = np.median(ev[m])
            pooled[side].append((j[ok], ev[ok]))
            log(f"{('port', 'starboard')[side]} pass {i}: {int(use.sum())} bins of {use.size} used, "
                f"beam angle 5-95% {np.degrees(np.percentile(b, 5)):.0f}..{np.degrees(np.percentile(b, 95)):.0f} deg")
    resp, count = np.full((2, len(KNOTS_DEG)), np.nan), np.zeros((2, len(KNOTS_DEG)), int)
    for side in (0, 1):
        j = np.concatenate([a for a, _ in pooled[side]])
        ev = np.concatenate([b for _, b in pooled[side]])
        for k in range(len(KNOTS_DEG)):
            m = j == k
            count[side, k] = m.sum()
            if m.sum() >= MIN_COUNT:
                resp[side, k] = np.median(ev[m])
    level = np.nanmax(resp, axis=1, keepdims=True)  # the constant goes to the gain
    per_pass = per_pass - level[None]
    resp = resp - level
    full = resp.copy()
    for side in (0, 1):  # angles without enough bins continue at the nearest measured one
        ok = np.isfinite(full[side])
        full[side] = np.interp(KNOTS_DEG, KNOTS_DEG[ok], full[side][ok])
        log(f"{('port', 'starboard')[side]}: measured {KNOTS_DEG[ok][0]:.1f}..{KNOTS_DEG[ok][-1]:.1f} deg; correction to the Gaussian "
            + " ".join(f"{a:.0f}:{v:+.2f}" for a, v in zip(KNOTS_DEG[ok][::3], full[side][ok][::3])))
        spread = np.nanstd(per_pass[:, side], axis=0)
        log(f"  spread between passes (sd of their medians): median {np.nanmedian(spread):.2f}, max {np.nanmax(spread):.2f} log units")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, beam_knots=np.radians(KNOTS_DEG), beam_log=full, measured=resp, per_pass=per_pass, count=count)
    log(f"wrote {args.out}")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    gauss = -4 * np.log(2) * (np.radians(KNOTS_DEG)[None] - np.asarray(th.beta0).reshape(-1, 1)) ** 2 / np.asarray(th.omega).reshape(-1, 1) ** 2
    gauss = np.broadcast_to(gauss, (2, len(KNOTS_DEG)))
    for side, ax in enumerate(axs):
        for i in range(len(cp)):
            ax.plot(KNOTS_DEG, per_pass[i, side], color="0.6", lw=0.8, label="each pass" if i == 0 else None)
        ax.plot(KNOTS_DEG, resp[side], "k", lw=2, label="all passes: correction to the Gaussian")
        ax.plot(KNOTS_DEG, gauss[side], "C0--", lw=1.2, label="the Gaussian beam held so far")
        ax.plot(KNOTS_DEG, gauss[side] + full[side] - np.max(gauss[side] + full[side]), "C3", lw=1.6, label="Gaussian plus correction")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set(title=("port", "starboard")[side] + ": response in beam angle", xlabel="beam angle below horizontal (deg)")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("log response")
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=110)
    log(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
