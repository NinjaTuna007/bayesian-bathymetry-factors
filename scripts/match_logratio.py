#!/usr/bin/env python3
r"""Sanity check of the match factor on real data: pair residuals of patches A and B against :math:`p_L`.

Usage: python scripts/match_logratio.py PATCHES.npz [--draws 4000] [--out figs/match_logratio.png]

The seabed is the diagnostic 3DSS seabed of both patches (`seabed_3dss`), not something
the method uses. Locations drawn in the overlap of the two passes are projected by both,
the pings involved are rendered, pairs are selected with the echo guard, and the
log-ratio residuals :math:`e - \sigma\hat\mu` are compared with :math:`p_L` and with residuals
simulated from the same pairs' predictions (one random reflectivity per pair). The prior
mean of :math:`\log\kappa` per side, which sets the noise floor of :math:`\bar J`, is the median
of :math:`\log(y / g S)` over the lit bins at the drawn locations.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE
from scipy.special import polygamma
from seabed_3dss import bathymetry_world, seabed_on_mesh

from bbf import match as Mt
from bbf import ping as P
from bbf.io import load_patch
from bbf.ping.factor import DB_TO_LOG_I

jax.config.update("jax_enable_x64", True)


def render_involved(mesh, d, poses, pings, theta, u, n_bins, dr) -> np.ndarray:
    """Unit intensities (n, 2, n_bins) of a pass, rendered only for the listed pings (zero elsewhere)."""
    S = np.zeros((len(poses), 2, n_bins))
    S[pings] = np.asarray(P.unit_intensities(mesh, jnp.asarray(d), poses[pings], theta, u, n_bins, dr))
    return S


def log_kappa_per_side(proj, S, y, dr) -> np.ndarray:
    r"""Median :math:`\log(y / g S)` per side over lit bins (y above 10 g N) at the projected draws."""
    ping, side, b = proj
    ok = ping >= 0
    ping, side, b = ping[ok], side[ok], b[ok]
    r = np.asarray(P.bin_ranges(y.shape[-1], dr))[b]
    g = np.exp(DB_TO_LOG_I * (GAIN_DB[side, 0] + GAIN_DB[side, 1] * r + GAIN_DB[side, 2] * np.log10(r)))
    yy, SS = y[ping, side, b], S[ping, side, b]
    lit = (SS > 0) & (yy > 10 * g * NOISE[side])
    return np.array([np.median(np.log(yy[lit & (side == c)] / (g * SS)[lit & (side == c)])) for c in (0, 1)])


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("figs/match_logratio.png"))
    args = ap.parse_args()

    A, B = load_patch(args.patches, "A"), load_patch(args.patches, "B")
    n_bins, dr = A.n_bins, A.dr
    z_ref = float(np.median(np.concatenate([A.poses[:, 2], B.poses[:, 2]])))
    mesh, d, seen = seabed_on_mesh(np.concatenate([bathymetry_world(args.patches, A),
                                                    bathymetry_world(args.patches, B)]), z_ref)
    pa, pb = A.local_poses(mesh.origin), B.local_poses(mesh.origin)
    print(f"mesh {mesh.n_vertices} vertices, {seen.mean():.0%} with 3DSS points")

    q = Mt.draw_overlap(pa, pb, n_bins * dr, args.draws, rng=0)
    proj_a = Mt.project(mesh, d, pa, q, n_bins, dr)
    proj_b = Mt.project(mesh, d, pb, q, n_bins, dr)
    both = (proj_a[0] >= 0) & (proj_b[0] >= 0)
    print(f"{len(q)} draws in the overlap, {both.sum()} seen by both passes")

    base = P.Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=0.0,
                   log_kappa_std=1.0, gain_db=GAIN_DB)
    u = P.u_grid(n_bins, dr)
    S_a = render_involved(mesh, d, pa, np.unique(proj_a[0][both]), base, u, n_bins, dr)
    S_b = render_involved(mesh, d, pb, np.unique(proj_b[0][both]), base, u, n_bins, dr)
    sub_a = tuple(x[both] for x in proj_a)
    sub_b = tuple(x[both] for x in proj_b)
    m_kappa = np.mean([log_kappa_per_side(sub_a, S_a, A.y, dr), log_kappa_per_side(sub_b, S_b, B.y, dr)], axis=0)
    theta = P.Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=m_kappa,
                    log_kappa_std=1.0, gain_db=GAIN_DB)
    print(f"log kappa prior mean per side: {m_kappa.round(2)}")

    pairs = Mt.select_pairs(proj_a, proj_b, S_a, S_b, A.y, B.y, theta, dr)
    e = np.asarray(Mt.residual(pairs, S_a, S_b, A.y, B.y, theta, dr))
    value, mu = Mt.log_match_factor(pairs, S_a, S_b, A.y, B.y, theta, dr)
    mu = float(mu)
    sign = pairs.sign
    e_c = e - sign * mu
    sd_ref = float(np.sqrt(2 * polygamma(1, L_SHAPE)))
    print(f"{len(pairs)} pairs ({(sign != 0).sum()} mix port and starboard); fitted mu = {mu:+.2f}; "
          f"log f_ab = {float(value):.1f}")
    print(f"e - sigma mu: mean {e_c.mean():+.2f}, std {e_c.std():.2f} (pure speckle {sd_ref:.2f} at L = {L_SHAPE}); "
          f"same-side mean {e[sign == 0].mean():+.2f}, mixed mean {e[sign != 0].mean():+.2f}")

    # reference: the same pairs' predictions, one random reflectivity per pair, gamma speckle
    rng = np.random.default_rng(1)
    reps = 50
    ka = np.tile(S_a[pairs.ping_a, pairs.side_a, pairs.bin_a], reps)
    kb = np.tile(S_b[pairs.ping_b, pairs.side_b, pairs.bin_b], reps)
    rho = np.exp(rng.normal(0.0, 2.0, ka.size))
    e_sim = np.log(rng.gamma(L_SHAPE, rho * kb / L_SHAPE)) - np.log(rng.gamma(L_SHAPE, rho * ka / L_SHAPE)) \
        - np.log(kb / ka)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.linspace(-12, 12, 400)
    bins = np.linspace(-12, 12, 97)
    ax.hist(e_sim, bins=bins.tolist(), density=True, histtype="step", color="0.5", label=f"simulated ({e_sim.size} pairs)")
    ax.hist(e_c, bins=bins[::2].tolist(), density=True, histtype="step", color="C3", lw=1.2,
            label=rf"measured $e - \sigma\hat\mu$, A vs B ({len(e)} pairs)")
    ax.plot(xs, np.exp(np.asarray(Mt.log_ratio_pdf(jnp.asarray(xs), L_SHAPE))), "k", lw=1, label=f"$p_L$, L = {L_SHAPE}")
    ax.set_xlabel(r"$e = \log(y'/y) - \log(\bar J'/\bar J)$")
    ax.set_title(rf"pair log-ratio residual, 3DSS seabed (diagnostic), $\hat\mu$ = {mu:+.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
