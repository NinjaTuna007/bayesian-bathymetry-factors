#!/usr/bin/env python3
r"""Per-bin table of measured and predicted intensities with the geometry behind each bin.

Usage: python scripts/residual_table.py PATCHES.npz DEPTHS.npz --out TABLE.npz [--k 9]

At the state in DEPTHS.npz (`bbf.io.load_depths`: depths, and :math:`\psi`, :math:`\xi` when it has
them), with bins averaged over :math:`k`, the script sets the problem up as `scripts/infer.py`
does, refits the spike parameters and the log-:math:`\kappa` prior, fits :math:`\xi` with the depths
held when the file has none (unless `--no-xi`), and writes for every bin of every ping-side

* the measured intensity :math:`y`, the predicted means :math:`\bar I`, :math:`\bar I_\mathrm{sp}` at
  the modes :math:`\hat u`, and whether the ping factor scores the bin;
* the unit intensity with the field, :math:`S`, and without, :math:`S_0`;
* the bin's ground: with :math:`\gamma_k = V_k \Delta q_k / \Delta r` the visible curve length of
  sample :math:`k`, the sums :math:`A = \sum_k \gamma_k w_b(r_k)` (and :math:`A_1`, the same without
  :math:`V`) and the :math:`\gamma`-weighted means of the range, the beam angle :math:`\beta`, the
  incidence cosine :math:`\cos\varphi`, the cosine on a level seabed, the footprint, :math:`\xi`
  and the seabed slope outboard and along track.

`scripts/residual_models.py` and `scripts/residual_anova.py` read the table. Only the
intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE

from bbf import mesh as M
from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.calibrate import calibrate
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import jacobians, labels_of
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.inference.objective import Problem, render, value_and_grad
from bbf.io import load_depths, load_patch
from bbf.match import Reflectivity
from bbf.ping import (
    Theta,
    bin_ranges,
    onset_valid,
    ping_use,
    predicted,
    u_grid,
    water_column_lit_rate,
    with_psi,
)
from bbf.ping import geometry as G
from bbf.ping.render import curve_length, split_into_bins

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

RHO, SIGMA, ALPHA = 5.0, 8.0, 3  # depth prior of scripts/infer.py
RHO_XI, SIGMA_XI = 2.0, 1.0
FEATURES = ("S", "S0", "A", "A1", "r", "beta", "cos", "cos_flat", "qx", "qy", "xi", "slope_out", "slope_fwd", "depth")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bin_features(mesh: M.Mesh, d, pose, side: int, theta: Theta, u, n_bins: int, dr: float, xi):
    r"""Per-bin sums and :math:`\gamma`-weighted means of one ping-side, stacked (len(FEATURES), n_bins); NaN means where :math:`A = 0`."""
    P, n, inside = G.samples(mesh, d, pose, side, u)
    s, R = G.pose_frame(pose)
    axes = R @ jnp.asarray(G.R_MOUNT[side])
    a, r = G.echo(theta, pose, side, P, n)
    e = (P - s) / r[:, None]
    beta = jnp.arctan2(e @ axes[:, 2], e @ axes[:, 1])
    cos_phi = jnp.maximum(-jnp.sum(n * e, axis=-1), 0.0)
    V = G.horizon_visibility(P, s, u, theta.shadow_margin, theta.wavelength, inside)
    wet = inside & (P[:, 2] <= theta.z_water)
    dq = jnp.where(wet, curve_length(P[:, :2]), 0.0) / dr
    gam = V * dq
    x = G.field_at(mesh, xi, P[:, :2])
    _, _, _, fwd, out = G.side_frame(pose, side)
    grad = -n[:, :2] / n[:, 2:3]  # gradient of the depth: the face normal is (g, 1) normalised
    cols = [r, beta, cos_phi, -e[:, 2], P[:, 0], P[:, 1], x, grad @ out, grad @ fwd, mesh.z_ref - P[:, 2]]

    def b(w):
        return split_into_bins(r, w, n_bins, dr)

    A = b(gam)
    means = [jnp.where(A > 0, b(gam * c) / jnp.where(A > 0, A, 1.0), jnp.nan) for c in cols]
    return jnp.stack([b(gam * a * jnp.exp(x)), b(gam * a), A, b(dq), *means])


def feature_table(mesh, d, poses, theta, u, n_bins, dr, xi, chunk: int = 32) -> np.ndarray:
    """`bin_features` of both sides of every pose: (n, 2, len(FEATURES), n_bins), float32."""
    f = jax.jit(lambda pp: jnp.stack(
        [jax.vmap(lambda pose, side=side: bin_features(mesh, d, pose, side, theta, u, n_bins, dr, xi))(pp)
         for side in (0, 1)], axis=1))
    out = np.zeros((len(poses), 2, len(FEATURES), n_bins), np.float32)
    for s in range(0, len(poses), chunk):
        idx = np.minimum(np.arange(s, s + chunk), len(poses) - 1)
        out[idx] = np.asarray(f(jnp.asarray(poses[idx])))
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("depths", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=2, help="rounds of the field fit when the depths file has no xi")
    ap.add_argument("--inner", type=int, default=6)
    ap.add_argument("--refit-xi", action="store_true", help="fit xi at the held depths even when the file has one")
    ap.add_argument("--no-xi", action="store_true", help="leave xi at zero: the geometry and S0 only, which is quick")
    ap.add_argument("--shift-b", type=float, nargs=2, default=[0.0, 0.0], metavar=("EAST", "NORTH"),
                    help="move pass B's positions by this much (m), to try a horizontal offset between the passes")
    args = ap.parse_args()

    passes = [load_patch(args.patches, n) for n in "AB"]
    if any(args.shift_b):
        moved = passes[1].poses.copy()
        moved[:, :2] += np.asarray(args.shift_b)
        passes[1] = dataclasses.replace(passes[1], poses=moved)
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    mesh, d, extra = load_depths(args.depths)
    beta = np.asarray(extra["beta"], float)
    if "psi" in extra:
        theta = with_psi(theta, extra["psi"])
    xi = None if args.refit_xi or "xi" not in extra else np.asarray(extra["xi"], float)
    if args.no_xi:
        xi = np.zeros(mesh.n_vertices)
    prior = PR.Prior(mesh, RHO, SIGMA, mu_beta=beta, Sigma_beta=np.array([[100.0]]) if len(beta) == 1 else
                     np.diag([0.04, 0.04, 25.0]), alpha=ALPHA)
    cp = [coarsen_pass(p, args.k) for p in passes]
    th = coarsen_theta(theta, args.k)
    dr, n_bins = cp[0].dr, cp[0].n_bins
    u = u_grid(n_bins, dr, du=max(0.005, dr / 3))
    labels = [labels_of(p, th) for p in cp]
    r0 = [I.first_echo(p, th) for p in cp]
    valid = [onset_valid(r, lab.shape[-1], th.block * dr) for r, lab in zip(r0, labels)]
    th = dataclasses.replace(th, wc_lit=water_column_lit_rate(labels, r0, th.block * dr))
    poses = [p.local_poses(mesh.origin) for p in cp]
    offsets = [p.log_power() for p in cp]
    base = Problem(mesh=mesh, prior=prior, theta=th, poses=poses, y=[p.y for p in cp], labels=labels, u=u, dr=dr,
                   chunk=32, valid=valid, refl=Reflectivity(mesh, RHO_XI, SIGMA_XI), kappa_offset=offsets)
    log(f"k = {args.k}: {n_bins} bins of {dr * 100:.1f} cm; state {args.depths.name}, "
        f"psi {'file' if 'psi' in extra else 'prototype'}, xi {'zero' if args.no_xi else 'file' if xi is not None else 'fitted here'}")

    def calibrated(xi_now):
        S = render(base, d, xi=xi_now)
        sub = [np.arange(0, len(p), 50) for p in poses]
        use_c = np.concatenate([np.stack([ping_use(lab[t], n_bins, th.block, valid=val[t]) for t in s])
                                for lab, val, s in zip(labels, valid, sub)])
        fitted, _ = calibrate(np.concatenate([p.y[s] for p, s in zip(cp, sub)]),
                              np.concatenate([Sp[s] for Sp, s in zip(S, sub)]), use_c, th, dr,
                              offset=np.concatenate([o[s] for o, s in zip(offsets, sub)]))
        return dataclasses.replace(base, theta=fitted)

    pb = calibrated(xi)
    if xi is None:
        xi = np.zeros(mesh.n_vertices)
        for _ in range(args.rounds):
            _, _, _, xi, _ = gauss_newton(pb, d, beta, jacobians(pb, d, xi), xi=xi, hold_depths=True,
                                          max_iter=args.inner, log=log)
        pb = calibrated(xi)
    tf = pb.theta
    _, _, det = value_and_grad(pb, d, beta, xi=xi)
    log(f"data term {det['data']:.0f}; xi sd {xi.std():.2f}, range {xi.min():+.1f} to {xi.max():+.1f}")

    feats = np.concatenate([feature_table(mesh, jnp.asarray(d), pp, tf, u, n_bins, dr, jnp.asarray(xi)) for pp in poses])
    u_hat = np.concatenate(det["u"])
    S = np.concatenate(det["S"])
    I_bar, I_sp = (np.stack([np.asarray(predicted(S[:, c], u_hat[:, c, None], tf, c, dr)[k]) for c in (0, 1)], axis=1)
                   for k in (0, 1))
    y = np.concatenate([p.y for p in cp])
    use = np.concatenate(pb.uses()) & (y > 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, Any] = {f"f_{name}": feats[:, :, i] for i, name in enumerate(FEATURES)}
    np.savez(args.out, **cols,
             y=y.astype(np.float32), I_bar=I_bar.astype(np.float32), I_sp=I_sp.astype(np.float32), use=use,
             lit=np.concatenate([np.repeat(lab, th.block, axis=-1) for lab in labels]),
             u_hat=u_hat, pass_id=np.concatenate([np.full(len(p), i) for i, p in enumerate(poses)]),
             ping=np.concatenate([np.arange(len(p)) for p in poses]), poses=np.concatenate(poses),
             t=np.concatenate([p.t for p in cp]), offset=np.concatenate(offsets), r0=np.concatenate(r0),
             r_bin=np.asarray(bin_ranges(n_bins, dr)), dr=dr, k=args.k, L=tf.L, noise=np.asarray(tf.noise),
             eps=np.asarray(tf.eps), eta=np.asarray(tf.eta), zeta=np.asarray(tf.zeta), ell=np.asarray(tf.ell),
             gain_db=np.asarray(tf.gain_db), beta0=np.asarray(tf.beta0), omega=np.asarray(tf.omega), alpha=tf.alpha,
             log_kappa_mean=np.asarray(tf.log_kappa_mean), log_kappa_std=np.asarray(tf.log_kappa_std),
             xi=xi, d=d, mesh_h=mesh.h, mesh_x0=mesh.x0, mesh_y0=mesh.y0, mesh_rows=mesh.n_rows, mesh_cols=mesh.n_cols,
             z_ref=mesh.z_ref, data=det["data"])
    log(f"wrote {args.out}: {int(use.sum())} scored bins of {use.size}")


if __name__ == "__main__":
    main()
