#!/usr/bin/env python3
r"""Status of the match on real data: what the shared reflectivity field does at the first-echo start.

Usage: python scripts/match_status.py PATCHES.npz [--k 9] [--rho 2.0] [--sigma 1.0] [--out figs/match_status.png]

At the start depths :math:`d_0` (first echoes, as `scripts/infer.py`), with bins averaged over :math:`k`:

1. the uniform-reflectivity model (one :math:`\kappa_{t,c}` per ping-side, :math:`\xi = 0`);
2. the log-reflectivity field fitted with the depths held (`gauss_newton(hold_depths=True)`), to
   both passes together, :math:`\hat\xi`, and to each alone, :math:`\hat\xi_A, \hat\xi_B`.

With :math:`F_p(d, \xi)` the data term of pass :math:`p`, the passes' agreement is scored by

.. math::

    M(d) = \max_\xi \bigl[F_A + F_B + \log p(\xi)\bigr]
         - \max_\xi \bigl[F_A + \log p(\xi)\bigr] - \max_\xi \bigl[F_B + \log p(\xi)\bigr],

the log evidence lost by making them share one reflectivity: the match factor of this model.
Its gradient in the depths is, at the three modes,
:math:`\nabla_d M = g(\hat\xi) - g_A(\hat\xi_A) - g_B(\hat\xi_B)` with :math:`g` the data gradients, and
its share of the data gradient :math:`\lVert \nabla_d M \rVert / \lVert g(\hat\xi) \rVert` is what the
passes' agreement contributes to the step in :math:`d`.

The fit between the passes is checked on ground cells (`bbf.match.cells`): the scores :math:`z`
of the cell residuals are standard normal when the passes agree up to speckle.

The sonar's transmit power steps from 10% to 50% inside pass A (at ping 147 of the cache's
patch; the cache does not hold the power yet). The step is common to a ping, so the ping-side
gains :math:`\kappa_{t,c}` take it up and the field does not see it; the fitted step is printed.

Uses only the intensities and the navigation; the 3DSS bathymetry is never read.
"""
import argparse
import dataclasses
import time
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE

from bbf import match as Mt
from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import jacobians, labels_of
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.inference.objective import Problem, value_and_grad
from bbf.io import load_patch
from bbf.ping import (
    Theta,
    bin_ranges,
    onset_valid,
    predicted,
    u_grid,
    water_column_lit_rate,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_field(pb: Problem, d, beta, rounds: int, inner: int):
    r""":math:`\hat\xi` at held depths, the data term's details there and :math:`F + \log p(\xi)`."""
    xi = np.zeros(pb.mesh.n_vertices)
    for _ in range(rounds):
        _, _, _, xi, _ = gauss_newton(pb, d, beta, jacobians(pb, d, xi), xi=xi, hold_depths=True, max_iter=inner, log=log)
    _, _, det = value_and_grad(pb, d, beta, xi=xi)
    assert pb.refl is not None
    return xi, det, det["data"] + pb.refl.log_density(xi)


def cell_check(pb: Problem, fp, S, u_hat, cell: float):
    r"""Cell residuals between passes 0 and 1: (centres (m, 2), :math:`e`, :math:`z`, range in each pass (m, 2))."""
    r = np.asarray(bin_ranges(pb.n_bins, pb.dr))
    sums, rng = [], []
    for p in (0, 1):
        Ib = np.stack([np.asarray(predicted(S[p][:, c], u_hat[p][:, c, None], pb.theta, c, pb.dr)[0]) for c in (0, 1)],
                      axis=1)
        use = pb.uses()[p] & (pb.y[p] > 0)
        sums.append(Mt.cell_sums(fp[p], pb.y[p], Ib, use, cell, pb.theta.L))
        rng.append(Mt.cell_sums(fp[p], r * Ib, Ib, use, cell, pb.theta.L))  # mean range under the predicted echo
    key, e, ka, kb = Mt.cell_residual(sums[0], sums[1])
    _, dlog, _, _ = Mt.cell_residual(rng[0], rng[1])
    ra = {tuple(k): v for k, v in zip(rng[0].key.tolist(), rng[0].y / rng[0].mean)}
    r_a = np.array([ra[tuple(k)] for k in key.tolist()])
    return (key + 0.5) * cell, e, Mt.ratio_score(e, ka, kb), np.column_stack([r_a, r_a * np.exp(dlog)])


def spread(z):
    q = np.percentile(z, [25, 50, 75])
    return f"median {q[1]:+.2f}, IQR {q[2] - q[0]:.2f}, |z| < 3 in {np.mean(np.abs(z) < 3):.0%}"


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--cell", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--inner", type=int, default=6)
    ap.add_argument("--power-step", type=int, default=147, help="first ping of pass A at 50%% transmit power")
    ap.add_argument("--out", type=Path, default=Path("figs/match_status.png"))
    args = ap.parse_args()

    passes = [load_patch(args.patches, n) for n in "AB"]
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    mesh = I.mesh_for_passes(passes, h=0.5, buffer=10.0)
    pts = np.concatenate([I.first_echo_points(p, I.first_echo(p, theta)) for p in passes])
    beta = I.plane_fit(pts, mesh)
    prior = PR.Prior(mesh, 5.0, 4.0, mu_beta=beta, Sigma_beta=np.diag([0.04, 0.04, 25.0]))
    d = I.depths_from_points(mesh, pts, prior)
    cp = [coarsen_pass(p, args.k) for p in passes]
    th = coarsen_theta(theta, args.k)
    dr = cp[0].dr
    labels = [labels_of(p, th) for p in cp]
    r0 = [I.first_echo(p, th) for p in cp]
    valid = [onset_valid(r, lab.shape[-1], th.block * dr) for r, lab in zip(r0, labels)]
    th = dataclasses.replace(th, wc_lit=water_column_lit_rate(labels, r0, th.block * dr))
    u = u_grid(cp[0].n_bins, dr, du=dr / 3)
    poses = [p.local_poses(mesh.origin) for p in cp]
    refl = Mt.Reflectivity(mesh, args.rho, args.sigma)
    plain = Problem(mesh=mesh, prior=prior, theta=th, poses=poses, y=[p.y for p in cp], labels=labels, u=u, dr=dr,
                    chunk=32, valid=valid)
    both = dataclasses.replace(plain, refl=refl)
    single = [dataclasses.replace(both, poses=[poses[p]], y=[cp[p].y], labels=[labels[p]], valid=[valid[p]])
              for p in (0, 1)]
    n_bins = int(sum(x.sum() for x in plain.uses()))
    log(f"k = {args.k}: {mesh.n_vertices} vertices, {n_bins} scored ping bins; field rho {args.rho} m, sigma {args.sigma}")

    # 1. uniform reflectivity
    _, _, det0 = value_and_grad(plain, d, beta)
    g0 = det0["grad_d_data"]
    fp = [Mt.footprints(mesh, d, pp, th, u, plain.n_bins, dr) for pp in poses]
    xy, e0, z0, rr = cell_check(plain, fp, det0["S"], det0["u"], args.cell)
    log(f"uniform reflectivity: data term {det0['data']:.0f}, |grad_d| {np.linalg.norm(g0):.3g}")
    log(f"  cells of {args.cell} m seen by both passes: {len(z0)}; score z: {spread(z0)}; "
        f"residual e IQR {np.subtract(*np.percentile(e0, [75, 25])):.2f}")

    # 2. the field, at held depths
    xi, det, ev = fit_field(both, d, beta, args.rounds, args.inner)
    fits = [fit_field(pb, d, beta, args.rounds, args.inner) for pb in single]
    g = det["grad_d_data"]
    g_sep = fits[0][1]["grad_d_data"] + fits[1][1]["grad_d_data"]
    # vertices each pass sees: those in a ground cell holding one of its footprints
    vcell = np.floor(mesh.vertices / args.cell).astype(np.int64)
    seen = []
    for f in fp:
        cells = {tuple(c) for c in np.unique(np.floor(f[np.isfinite(f).all(axis=-1)] / args.cell).astype(np.int64), axis=0).tolist()}
        seen.append(np.array([tuple(c) in cells for c in vcell.tolist()]))
    ov = seen[0] & seen[1]
    M = ev - fits[0][2] - fits[1][2]
    _, e1, z1, _ = cell_check(both, fp, det["S"], det["u"], args.cell)
    log(f"shared field: data term {det['data']:.0f} (+{det['data'] - det0['data']:.0f}), xi sd {xi[seen[0] | seen[1]].std():.2f} "
        f"(range {xi.min():+.1f} to {xi.max():+.1f}), "
        f"left |grad_xi| {np.linalg.norm(det['grad_xi']):.3g}")
    log(f"  cell score z: {spread(z1)}; residual e IQR {np.subtract(*np.percentile(e1, [75, 25])):.2f}")
    log(f"  match evidence M = {M:.0f} nats ({M / max(len(z1), 1):.1f} per shared cell)")
    log(f"  |grad_d| data {np.linalg.norm(g):.3g}; passes apart {np.linalg.norm(g_sep):.3g}; "
        f"match (grad M) {np.linalg.norm(g - g_sep):.3g} = {np.linalg.norm(g - g_sep) / np.linalg.norm(g):.1%} of the data gradient "
        f"({np.linalg.norm((g - g_sep)[ov]) / np.linalg.norm(g[ov]):.1%} over the {ov.sum()} vertices both passes see)")
    # pass A's transmit power steps from 10% to 50% at this ping: the ping-side gains take the step up
    ua, k = det["u"][0], args.power_step
    log(f"  log kappa of A, pings from {k} minus those before: port {ua[k:, 0].mean() - ua[:k, 0].mean():+.2f}, "
        f"starboard {ua[k:, 1].mean() - ua[:k, 1].mean():+.2f} (log 5 = {np.log(5):.2f})")
    dxi = fits[1][0] - fits[0][0]
    for name, (x, _, _) in zip("AB", fits):
        log(f"  field of {name} alone: range {x.min():+.1f} to {x.max():+.1f}")
    log(f"  per-pass fields differ over the overlap by sd {dxi[ov].std():.2f} (median {np.median(dxi[ov]):+.2f})")

    # figure
    v = mesh.vertices
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    es = np.linspace(-6, 6, 97)
    ax[0, 0].hist(e0, bins=es.tolist(), density=True, histtype="step", color=BLUE, lw=2, label="uniform reflectivity")
    ax[0, 0].hist(e1, bins=es.tolist(), density=True, histtype="step", color=ORANGE, lw=2, label="shared field")
    ax[0, 0].axvline(0, color=INK, lw=1)
    ax[0, 0].set(xlabel="cell residual e (log); speckle alone is about 0.1 wide", ylabel="density",
                 title=f"Pass A against B on {args.cell:g} m ground cells")
    ax[0, 0].legend(frameon=False)
    for a, e, name in ((ax[0, 1], e0, "uniform reflectivity"), (ax[0, 2], e1, "shared field")):
        sc = a.scatter(rr[:, 0], rr[:, 1], c=e, s=3, cmap=DIVERGING, vmin=-3, vmax=3, linewidths=0)
        fig.colorbar(sc, ax=a, label="cell residual e (log)")
        a.set(xlabel="range in A (m)", ylabel="range in B (m)", title=f"Residual by where each pass saw the cell: {name}")
    lim = 2 * args.sigma
    sn = seen[0] | seen[1]
    for a, val, mask, title in ((ax[1, 0], xi, sn, r"Shared log reflectivity $\hat\xi$"),
                                (ax[1, 1], dxi, ov, r"Disagreement of the passes' own fields, $\hat\xi_B - \hat\xi_A$")):
        sc = a.scatter(v[mask, 0], v[mask, 1], c=val[mask], s=1, cmap=DIVERGING, vmin=-lim, vmax=lim, linewidths=0)
        fig.colorbar(sc, ax=a, label="log units")
        a.set(title=title)
    sc = ax[1, 2].scatter(xy[:, 0], xy[:, 1], c=e1, s=4, cmap=DIVERGING, vmin=-3, vmax=3, linewidths=0, marker="s")
    fig.colorbar(sc, ax=ax[1, 2], label="cell residual e (log)")
    ax[1, 2].set(title="What the shared field leaves between the passes")
    for a in ax[1]:
        for pp, name in zip(poses, "AB"):
            a.plot(pp[:, 0], pp[:, 1], color=INK, lw=1)
            a.annotate(name, pp[0, :2], color=INK)
        a.set(xlabel="east (m)", ylabel="north (m)", aspect="equal")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
