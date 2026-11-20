#!/usr/bin/env python3
r"""Range profile of one ping under the bbf ping factor, rebuilt from the prototype's figure.

Usage: python scripts/ping_profile.py PATCHES.npz [--patch A] [--ping 600 ...] [--out-dir figs]

PATCHES.npz is the prototype's patch cache (``cache/<bag>_patches.npz``: raw port and
starboard intensities, interpolated navigation and the 3DSS bathymetry per ping).

1. Seabed: the diagnostic 3DSS seabed of `seabed_3dss`: the 3DSS points, moved to the world frame, are
   averaged onto the mesh vertices; vertices without points get the conditional mean of the
   GMRF prior given the others. Depths are measured down from :math:`z_\mathrm{ref}`, the
   median sonar height.
2. :math:`\theta`: :math:`L`, :math:`N_c`, :math:`\epsilon_c` and the gain are the
   prototype's calibrated values. :math:`\eta_c`, :math:`\zeta_c`, :math:`\ell_c` and the prior
   on :math:`\log\kappa` come from `bbf.inference.calibrate` over the lit bins of every 100th
   ping, at the fixed seabed.
3. Figure: measured and predicted intensity along range for both sides of one ping, with
   :math:`\bar I` at the Laplace mode of :math:`\log\kappa`.
"""
import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE
from seabed_3dss import bathymetry_world, seabed_on_mesh

from bbf import ping as P
from bbf.inference.calibrate import calibrate
from bbf.io import load_depths, load_patch

jax.config.update("jax_enable_x64", True)

FIT_EVERY = 100


def render_pings(mesh, d, X, idx, n_bins, dr, theta, u):
    """Unit intensities (len(idx), 2, n_bins) at the fixed seabed, shadows included."""
    S = np.zeros((len(idx), 2, n_bins))
    dj = jnp.asarray(d)
    render = jax.jit(lambda pose, side: P.unit_intensity(mesh, dj, pose, side, theta, u, n_bins, dr),
                     static_argnums=1)
    for k, i in enumerate(idx):
        for side in (P.PORT, P.STARBOARD):
            S[k, side] = np.asarray(render(jnp.asarray(X[i]), side))
    return S


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--patch", default="A")
    ap.add_argument("--ping", type=int, nargs="+", default=[600], help="pings to plot, one figure each")
    ap.add_argument("--out-dir", type=Path, default=Path("figs"), help="writes ping_profile_<ping>.png")
    ap.add_argument("--depths", type=Path, help="seabed from a save_depths file (e.g. an inference result) "
                    "instead of the diagnostic 3DSS seabed")
    args = ap.parse_args()

    ps = load_patch(args.patches, args.patch)
    n_bins, dr = ps.n_bins, ps.dr
    r_bin = np.asarray(P.bin_ranges(n_bins, dr))
    z_ref = float(np.median(ps.poses[:, 2]))

    if args.depths is not None:
        mesh, d, _ = load_depths(args.depths)
        source = args.depths.stem
        print(f"seabed from {args.depths}: {mesh.n_vertices} vertices, z_ref = {mesh.z_ref:.2f}")
    else:
        mesh, d, seen = seabed_on_mesh(bathymetry_world(ps), z_ref)
        source = "3DSS seabed"
        print(f"mesh {mesh.n_vertices} vertices, {seen.mean():.0%} with 3DSS points; "
              f"depth {np.percentile(d[seen], 1):.1f}–{np.percentile(d[seen], 99):.1f} m below z_ref = {z_ref:.2f}")
    X_local = ps.local_poses(mesh.origin)

    idx = np.arange(0, ps.n_pings, FIT_EVERY)
    idx = np.union1d(idx, args.ping)
    y = ps.y[idx]
    u = P.u_grid(n_bins, dr)
    base = P.Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=0.0,
                   log_kappa_std=1.0, gain_db=GAIN_DB)
    S = render_pings(mesh, d, X_local, idx, n_bins, dr, base, u)

    labels = np.stack([[P.block_labels(y[k, c], base, c, dr) for c in (0, 1)] for k in range(len(idx))])
    use = np.stack([P.ping_use(labels[k], n_bins, base.block) for k in range(len(idx))])
    wc = np.median(y[:, :, (r_bin > 6) & (r_bin < 12)], axis=(0, 2))
    start = dataclasses.replace(base, eta=100 * wc, zeta=1e4 * wc, ell=np.full(2, 2.0))
    theta, fit = calibrate(y, S, use, start, dr)
    print(f"fit over {len(idx)} pings (converged {fit['converged']}): "
          f"eta {theta.eta}, zeta {theta.zeta}, ell {theta.ell} m, "
          f"log kappa {theta.log_kappa_mean} ± {theta.log_kappa_std}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in args.ping:
        out = args.out_dir / f"ping_profile_{i}.png"
        plot_ping(int(np.flatnonzero(idx == i)[0]), i, y, S, r_bin, dr, theta, out, source)
        print(f"wrote {out}")


def plot_ping(k, i, y, S, r_bin, dr, theta, out, source):
    r"""Measured, predicted and spike means along range, both sides of ping i (row k of y, S).

    :math:`\hat\kappa` is the Laplace mode over the bins the ping factor scores (lit blocks);
    blocks labelled shadow are shaded.
    """
    n_bins = y.shape[-1]
    labels = np.stack([P.block_labels(y[k, c], theta, c, dr) for c in (0, 1)])
    use = P.ping_use(labels, n_bins, theta.block)
    I_bar, I_sp = np.zeros((2, n_bins)), np.zeros((2, n_bins))
    for c in (0, 1):
        u_hat, _ = P.mode(theta, c, dr, jnp.asarray(y[k, c]), jnp.asarray(S[k, c]), jnp.asarray(use[c]))
        I_bar[c], I_sp[c] = (np.asarray(v) for v in P.predicted(jnp.asarray(S[k, c]), u_hat, theta, c, dr))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for c, name in ((P.PORT, "port"), (P.STARBOARD, "starboard")):
        ax = axs[c]
        ax.semilogy(r_bin, np.maximum(y[k, c], 1), lw=0.3, color="0.6", label="measured $y$")
        ax.semilogy(r_bin, I_bar[c], color="C3", lw=1.2, label=r"predicted $\bar I = g(\hat\kappa S + N)$")
        ax.semilogy(r_bin, I_sp[c], color="C1", lw=0.8, ls=":", label="spike component mean")
        ax.fill_between(r_bin, 0, 1, where=~use[c], transform=ax.get_xaxis_transform(), color="0.85",
                        step="mid", lw=0, zorder=0, label="labelled shadow")
        first = np.flatnonzero(S[k, c] > 0)
        if len(first):
            ax.axvline(r_bin[first[0]], color="C0", ls="--", lw=0.8, label=f"rendered first return ({source})")
        ax.set_title(f"ping {i}, {name}")
        ax.set_xlabel("slant range (m)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
