#!/usr/bin/env python3
r"""Range profile of one ping under the bbf ping factor, rebuilt from the prototype's figure.

Usage: python scripts/ping_profile.py PATCHES.npz [--patch A] [--ping 600 ...] [--out-dir figs]

PATCHES.npz is the prototype's patch cache (``cache/<bag>_patches.npz``: raw port and
starboard intensities, interpolated navigation and the 3DSS bathymetry per ping).

1. Seabed: the 3DSS bathymetry points, moved to the world frame with each ping's pose, are
   averaged onto the mesh vertices; vertices without points get the conditional mean of the
   GMRF prior given the others. Depths are measured down from :math:`z_\mathrm{ref}`, the
   median sonar height.
2. :math:`\theta`: :math:`L`, :math:`N_c`, :math:`\epsilon_c` and the gain are the
   prototype's calibrated values. :math:`\eta_c`, :math:`\zeta_c`, :math:`\ell_c` are fitted
   by maximum likelihood over every 100th ping together with one :math:`\log\kappa` per ping
   and side, at the fixed seabed; the prior on :math:`\log\kappa` is set from
   those fits.
3. Figure: measured and predicted intensity along range for both sides of one ping, with
   :math:`\bar I` at the Laplace mode of :math:`\log\kappa`.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.io import Pass, load_patch

jax.config.update("jax_enable_x64", True)

# Calibrated in the prototype (scripts/match_factor.py THETA_*, patch A, 2026-11-18).
GAIN_DB = np.array([[0.0, 0.65, 0.0], [0.0, 0.0, 4.1]])
L_SHAPE = 0.48
NOISE = np.array([9.33e8, 5.28e7])
EPS = np.array([0.055, 0.021])

H_MESH, BUFFER = 0.5, 5.0
RHO, SIGMA, SIGMA_OBS = 3.0, 2.0, 0.05  # fill-in prior: range, std, per-point noise (m)
FIT_EVERY = 100


def bathymetry_world(ps: Pass, every: int = 2) -> np.ndarray:
    """3DSS points of every `every`-th ping in the world frame, (m, 3)."""
    frames = jax.vmap(P.pose_frame)(jnp.asarray(ps.poses))
    s, R = np.asarray(frames[0]), np.asarray(frames[1])
    keep = ps.bathy_ping % every == 0
    k = ps.bathy_ping[keep]
    return s[k] + np.einsum("mij,mj->mi", R[k], ps.bathy_xyz[keep])


def seabed_on_mesh(world: np.ndarray, z_ref: float) -> tuple[M.Mesh, np.ndarray, np.ndarray]:
    """Mesh over the points, depths from the points, and the vertices that had points."""
    lo, hi = world[:, :2].min(0), world[:, :2].max(0)
    mesh = M.build(lo[0], hi[0], lo[1], hi[1], h=H_MESH, buffer=BUFFER, z_ref=z_ref)
    q = mesh.to_local(world[:, :2])
    tri, bary, inside = (np.asarray(a) for a in M.locate(mesh, q))
    vert = mesh.triangles[tri, np.argmax(bary, axis=1)]
    depth = z_ref - world[:, 2]
    # drop points far off a robust plane (water-column detections)
    A = np.column_stack([q, np.ones(len(q))])
    coef = np.linalg.lstsq(A[inside], depth[inside], rcond=None)[0]
    res = depth - A @ coef
    mad = np.median(np.abs(res[inside]))
    ok = inside & (np.abs(res) < 10 * mad + 0.5)
    n = np.bincount(vert[ok], minlength=mesh.n_vertices).astype(float)
    mean = np.bincount(vert[ok], depth[ok], minlength=mesh.n_vertices) / np.maximum(n, 1)
    # conditional mean of the GMRF given the vertex averages
    prior = PR.Prior(mesh, RHO, SIGMA, mu_beta=coef, Sigma_beta=np.eye(3))
    W = sp.diags(n / SIGMA_OBS**2)
    rhs = prior.Lambda @ (prior.X @ coef) + W @ mean
    d = np.asarray(spla.spsolve((prior.Lambda + W).tocsc(), rhs))
    return mesh, d, n > 0


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


def fit_spikes(y, S, r, theta0):
    r"""ML fit of :math:`\log\eta_c, \log\zeta_c, \log\ell_c` and one :math:`\log\kappa` per ping and side."""
    m = y.shape[0]
    ok = y > 0
    ys = np.where(ok, y, 1.0)
    logg = np.stack([np.asarray(P.log_gain(r, GAIN_DB[c])) for c in (0, 1)])
    yj, Sj, okj, gj, rj = (jnp.asarray(a) for a in (ys, S, ok, np.exp(logg), r))

    def nll(x):
        log_eta, log_zeta, log_ell, u = x[0:2], x[2:4], x[4:6], x[6:].reshape(m, 2)
        I_bar = gj * (jnp.exp(u)[..., None] * Sj + jnp.asarray(NOISE)[:, None])
        extra = jnp.exp(log_eta)[:, None] + jnp.exp(log_zeta)[:, None] * jnp.exp(-rj / jnp.exp(log_ell)[:, None])
        I_sp = I_bar + gj * extra
        lp = jnp.logaddexp(jnp.log1p(-jnp.asarray(EPS))[:, None] + P.log_gamma(yj, L_SHAPE, I_bar),
                           jnp.log(jnp.asarray(EPS))[:, None] + P.log_gamma(yj, 1.0, I_sp))
        return -jnp.sum(jnp.where(okj, lp, 0.0))

    f = jax.jit(jax.value_and_grad(nll))
    sig = (S > 0) & ok
    u0 = np.array([[np.log(np.median(ys[k, c][sig[k, c]] / np.exp(logg[c])[sig[k, c]])
                           / np.median(S[k, c][sig[k, c]])) if sig[k, c].any() else 0.0
                    for c in (0, 1)] for k in range(m)])
    x0 = np.concatenate([np.log(theta0["eta"]), np.log(theta0["zeta"]), np.log(theta0["ell"]), u0.ravel()])
    res = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in f(jnp.asarray(x))), x0, jac=True,
                   method="L-BFGS-B", options={"maxiter": 2000})
    x = res.x
    return {"eta": np.exp(x[0:2]), "zeta": np.exp(x[2:4]), "ell": np.exp(x[4:6]), "u": x[6:].reshape(m, 2),
            "nll": float(res.fun), "converged": bool(res.success)}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--patch", default="A")
    ap.add_argument("--ping", type=int, nargs="+", default=[600], help="pings to plot, one figure each")
    ap.add_argument("--out-dir", type=Path, default=Path("figs"), help="writes ping_profile_<ping>.png")
    args = ap.parse_args()

    ps = load_patch(args.patches, args.patch)
    n_bins, dr = ps.n_bins, ps.dr
    r_bin = np.asarray(P.bin_ranges(n_bins, dr))
    z_ref = float(np.median(ps.poses[:, 2]))

    world = bathymetry_world(ps)
    mesh, d, seen = seabed_on_mesh(world, z_ref)
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

    wc = np.median(y[:, :, (r_bin > 6) & (r_bin < 12)], axis=(0, 2))
    fit = fit_spikes(y, S, r_bin, {"eta": 100 * wc, "zeta": 1e4 * wc, "ell": np.full(2, 2.0)})
    u_fit = fit["u"]
    theta = P.Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=fit["eta"], zeta=fit["zeta"], ell=fit["ell"],
                    log_kappa_mean=u_fit.mean(0), log_kappa_std=np.maximum(u_fit.std(0), 0.5), gain_db=GAIN_DB)
    print(f"fit over {len(idx)} pings (converged {fit['converged']}): "
          f"eta {theta.eta}, zeta {theta.zeta}, ell {theta.ell} m, "
          f"log kappa {theta.log_kappa_mean} ± {theta.log_kappa_std}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in args.ping:
        out = args.out_dir / f"ping_profile_{i}.png"
        plot_ping(int(np.flatnonzero(idx == i)[0]), i, y, S, r_bin, dr, theta, out)
        print(f"wrote {out}")


def plot_ping(k, i, y, S, r_bin, dr, theta, out):
    """Measured, predicted and spike means along range, both sides of ping i (row k of y, S)."""
    n_bins = y.shape[-1]
    I_bar, I_sp = np.zeros((2, n_bins)), np.zeros((2, n_bins))
    for c in (0, 1):
        u_hat, _ = P.mode(theta, c, dr, jnp.asarray(y[k, c]), jnp.asarray(S[k, c]), jnp.ones(n_bins, bool))
        I_bar[c], I_sp[c] = (np.asarray(v) for v in P.predicted(jnp.asarray(S[k, c]), u_hat, theta, c, dr))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for c, name in ((P.PORT, "port"), (P.STARBOARD, "starboard")):
        ax = axs[c]
        ax.semilogy(r_bin, np.maximum(y[k, c], 1), lw=0.3, color="0.6", label="measured $y$")
        ax.semilogy(r_bin, I_bar[c], color="C3", lw=1.2, label=r"predicted $\bar I = g(\hat\kappa S + N)$")
        ax.semilogy(r_bin, I_sp[c], color="C1", lw=0.8, ls=":", label="spike component mean")
        first = np.flatnonzero(S[k, c] > 0)
        if len(first):
            ax.axvline(r_bin[first[0]], color="C0", ls="--", lw=0.8, label="rendered first return (3DSS seabed)")
        ax.set_title(f"ping {i}, {name}")
        ax.set_xlabel("slant range (m)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
