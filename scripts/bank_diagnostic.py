#!/usr/bin/env python3
r"""What holds the depths off the bank in the first pass's starboard swath?

Usage: python scripts/bank_diagnostic.py PATCHES.npz STATE.npz [--k 27] [--refit] [--out runs/bank.json]

At a saved state of `scripts/infer.py` (depths, field, sonar parameters) and one level, the script
sets the problem up as the run does and reports, in bands of across-track distance from the
first pass's track on either side:

1. the gradient of each term in the depths (intensities: ping and shadow factors; first-return
   factor; prior), as the mean over the band's vertices. Depth is positive down, so a negative
   gradient asks for a shallower seabed;
2. the Gauss--Newton step of the depths there: nearly undamped (:math:`\lambda = 10^{-3}`) and as the
   trust region takes it, with the gain the model predicts for the latter;
3. the change the bank would need: the start that honours both sides' first echoes
   (`bbf.inference.init.bank_points`) minus the state's depths;
4. the bins of the first pass between each side's first echo and the model's first return there:
   how bright they are, how the mixture scores them, and their share of :math:`|\partial S / \partial d|`;
5. every term along the straight path from the state's depths to that bank start, with the field
   held (and, with `--refit`, after refitting the field at the path's end): whether the objective
   prefers the bank, and whether a barrier lies between.

Only the intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
import scipy.sparse as sp
import tvg_theta as TVG
from compare_runs import ALPHA, RHO, RHO_XI, SIGMA, SIGMA_XI, load_state, moved
from scipy.spatial import KDTree
from sksparse.cholmod import cho_factor

from bbf import prior as PR
from bbf.inference import ggn
from bbf.inference import init as I
from bbf.inference.gauss_newton import D_MAX_STEP, gauss_newton, predicted_gain
from bbf.inference.loop import infer, jacobians
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.inference.objective import data_term, value_and_grad
from bbf.io import load_patch
from bbf.match import Reflectivity
from bbf.ping import bin_ranges, log_gamma, predicted, u_grid
from bbf.ping.first_return import FirstReturn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

BANDS = [(-30.0, -20.0), (-20.0, -14.0), (-14.0, -8.0), (-8.0, -2.0), (-2.0, 2.0), (2.0, 8.0), (8.0, 14.0), (14.0, 20.0),
         (20.0, 30.0)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def across_track(mesh, ps) -> tuple[np.ndarray, np.ndarray]:
    """Per vertex: signed across-track distance from the pass's track (starboard positive) and whether it lies beside the track."""
    v = mesh.vertices + np.asarray(mesh.origin)
    i = KDTree(ps.poses[:, :2]).query(v)[1]
    yaw = ps.poses[i, 5]
    rel = v - ps.poses[i, :2]
    s = rel[:, 0] * np.cos(yaw) - rel[:, 1] * np.sin(yaw)  # starboard is :math:`(\cos\psi, -\sin\psi)`
    along = rel[:, 0] * np.sin(yaw) + rel[:, 1] * np.cos(yaw)
    beside = (np.abs(along) < 1.0) & (i > 0) & (i < ps.n_pings - 1)
    return s, beside


def pre_echo_bins(pb, J, det, r0) -> dict:
    """Per side of the first pass: the scored bins from its first echo r0 (n, 2) up to the model's first return."""
    th, n = pb.theta, pb.mesh.n_vertices
    r = np.asarray(bin_ranges(pb.n_bins, pb.dr))
    S, u_hat, use, y = det["S"][0], det["u"][0], pb.uses()[0], pb.y[0]
    off = np.zeros(len(y)) if pb.kappa_offset is None else np.asarray(pb.kappa_offset[0])
    out = {}
    for c, side in enumerate(("port", "stbd")):
        lit = S[:, c] > 1e-6 * S[:, c].max(axis=1, keepdims=True)
        first = np.where(lit.any(axis=1), r[np.argmax(lit, axis=1)], np.inf)  # range of the model's first return
        I_bar, I_sp = (np.asarray(a) for a in predicted(S[:, c], u_hat[:, c, None], th, c, pb.dr, off[:, None]))
        ys = np.where(y[:, c] > 0, y[:, c], 1.0)
        lm = np.log1p(-th.at(c).eps) + np.asarray(log_gamma(ys, th.L, I_bar))
        ls = np.log(th.at(c).eps) + np.asarray(log_gamma(ys, 1.0, I_sp))
        lp = np.logaddexp(lm, ls)
        ok = use[:, c] & (y[:, c] > 0)
        m = ok & (r[None] >= r0[:, c][:, None]) & (r[None] < first[:, None])
        jac = np.stack([np.asarray(abs(J[0][t][c][:, :n]).sum(axis=1)).ravel() for t in range(len(y))])
        if not m.any():
            out[side] = {"bins": 0}
            continue
        # no echo is predicted there, so the predicted mean is the background with its gain
        out[side] = {"bins": int(m.sum()), "share": float(m.sum() / max(ok.sum(), 1)),
                     "db_over_background": float(10 * np.log10(np.mean(y[:, c][m] / I_bar[m]))),
                     "spike_resp": float(np.mean(np.exp(ls - lp)[m])), "ll_bin": float(lp[m].mean()),
                     "ll_bin_rest": float(lp[ok & ~m].mean()), "jac_share": float(jac[m].sum() / max(jac[ok].sum(), 1e-300))}
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("state", type=Path)
    ap.add_argument("--k", type=int, default=27)
    ap.add_argument("--refit", action="store_true", help="refit the field at the bank start (6 steps)")
    ap.add_argument("--out", type=Path, default=Path("runs/bank.json"))
    args = ap.parse_args()
    raw = [load_patch(args.patches, n) for n in "AB"]
    st = load_state("state", args.state, len(raw))
    passes = moved(raw, st)
    mesh = st.mesh
    r0 = [I.first_echo(p, st.base) for p in passes]
    prior = PR.Prior(mesh, RHO, SIGMA, mu_beta=st.beta, Sigma_beta=np.array([[100.0]]), alpha=ALPHA)
    cp = [coarsen_pass(p, args.k) for p in passes]
    th = st.theta if st.k == args.k else coarsen_theta(st.base, args.k)  # the level's own fitted parameters when it is the state's
    u = u_grid(cp[0].n_bins, cp[0].dr, du=max(0.005, cp[0].dr / 3))
    out = infer(cp, mesh, prior, th, st.d, st.beta, u, rounds=1, inner=0, calib_every=50, chunk=32,
                refl=Reflectivity(mesh, RHO_XI, SIGMA_XI), xi0=st.xi, first_return=FirstReturn(r0), min_range=TVG.MIN_RANGE,
                fit_shape=st.k != args.k, mode_grid=49, shadow_stop=False, log=log)
    pb = out["problem"]
    d, beta, xi = st.d, st.beta, st.xi
    F = data_term(pb)
    ell, (gd, _), det = value_and_grad(pb, d, beta, F, xi=xi)
    g_data = det["grad_d_data"]
    g_prior = pb.prior.grad(d, beta)[0]
    g_fr = gd - g_data - g_prior
    log(f"state: ell {ell:.0f} = intensities {det['data']:.0f} + first return {det['first_return_value']:.0f} + priors")

    # the Gauss-Newton step of the depths: nearly undamped, and inside the trust region
    lay = ggn.layout(pb)
    n = lay["n"]
    J = jacobians(pb, d, xi)
    H = ggn.assemble(pb, J, det["S"], det["u"], S0=det["S"], fr=det["first_return"])
    g = np.zeros(lay["size"])
    g[:n], g[lay["beta"]:lay["beta"] + lay["n_beta"]] = gd, 0.0
    g[lay["xi"]:lay["xi"] + n] = det["grad_xi"]
    diag = np.maximum(H.diagonal(), 1e-12)
    raw_step = cho_factor(sp.csc_matrix(H + sp.diags(1e-3 * diag))).solve(g)
    prior_diag = pb.prior.joint_precision().diagonal()[:n]
    reach = diag[:n] > 2.0 * prior_diag
    scale = diag.copy()
    scale[:n] = np.maximum(np.maximum(diag[:n], np.median(diag[:n][reach])), np.abs(g[:n]) / D_MAX_STEP)
    lam, trust = 1.0, raw_step
    for _ in range(30):
        trust = cho_factor(sp.csc_matrix(H + sp.diags(lam * scale))).solve(g)
        if np.max(np.abs(trust[:n])) <= D_MAX_STEP:
            break
        lam *= 2.0
    log(f"Gauss-Newton step: nearly undamped max |dd| {np.abs(raw_step[:n]).max():.1f} m, predicted gain "
        f"{predicted_gain(H, g, raw_step, lay):.0f}; in the trust region (lambda {lam:.1f}) max |dd| "
        f"{np.abs(trust[:n]).max():.2f} m, predicted gain {predicted_gain(H, g, trust, lay):.0f}")

    # the bank start: both sides' first echoes
    pts = np.concatenate([I.first_echo_points(p, r) for p, r in zip(passes, r0)] + [I.bank_points(p, r) for p, r in zip(passes, r0)])
    d_bank = I.depths_from_points(mesh, pts, prior)
    need = d_bank - d
    s, beside = across_track(mesh, passes[0])
    rows = []
    log("band (m across A's track, starboard +): vertices | grad intensities, first return, prior (mean per vertex) | "
        "GN step raw, trust (mean, m) | to the bank start (mean, m)")
    for lo, hi in BANDS:
        m = beside & (s >= lo) & (s < hi)
        row = {"band": [lo, hi], "vertices": int(m.sum()), "grad_data": float(g_data[m].mean()), "grad_fr": float(g_fr[m].mean()),
               "grad_prior": float(g_prior[m].mean()), "step_raw": float(raw_step[:n][m].mean()),
               "step_raw_abs": float(np.abs(raw_step[:n][m]).mean()), "step_trust": float(trust[:n][m].mean()),
               "need": float(need[m].mean()), "diag_data": float((diag[:n][m] - prior_diag[m]).mean()),
               "cos_step_need": float(raw_step[:n][m] @ need[m] / (np.linalg.norm(raw_step[:n][m]) * np.linalg.norm(need[m]) + 1e-12))}
        rows.append(row)
        log(f"  {lo:+5.0f}..{hi:+5.0f}: {row['vertices']:5d} | {row['grad_data']:+9.2f} {row['grad_fr']:+9.2f} {row['grad_prior']:+9.2f} | "
            f"{row['step_raw']:+7.2f} (|.| {row['step_raw_abs']:.2f}) {row['step_trust']:+6.3f} | {row['need']:+6.2f}; "
            f"data curvature {row['diag_data']:.1f}; cos(step, need) {row['cos_step_need']:+.2f}")

    # the first pass's bins between each side's first echo and the model's first return: what the model makes of them
    pre = pre_echo_bins(pb, J, det, r0[0])
    for side, row in pre.items():
        log(f"A {side}: {row['bins']} scored bins between the first echo and the model's first return "
            f"({row['share']:.1%} of the side's scored bins), {row['db_over_background']:+.0f} dB over the background; "
            f"spike component takes {row['spike_resp']:.0%} of them; log density per bin {row['ll_bin']:+.1f} "
            f"against {row['ll_bin_rest']:+.1f} elsewhere; they hold {row['jac_share']:.1e} of the side's |dS/dd|")
    # the path to the bank start
    path = []
    log("path from the state to the bank start (field held): t | intensities, first return, depth prior, total (minus the state's)")
    base = None
    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        dt = d + t * need
        e, _, de = value_and_grad(pb, dt, beta, F, xi=xi)
        terms = np.array([de["data"], de["first_return_value"], pb.prior.log_density(dt, beta), e])
        base = terms if base is None else base
        rel = terms - base
        path.append({"t": t, "data": rel[0], "first_return": rel[1], "prior_d": rel[2], "total": rel[3]})
        log(f"  t = {t:.2f}: {rel[0]:+11.0f} {rel[1]:+9.0f} {rel[2]:+9.0f} {rel[3]:+11.0f}")
    result = {"bands": rows, "path": path, "lambda_trust": lam, "pre_echo": pre}
    if args.refit:
        _, _, _, xi_b, _ = gauss_newton(pb, d_bank, beta, jacobians(pb, d_bank, xi), xi=xi, hold_depths=True, max_iter=6, log=log)
        e, _, de = value_and_grad(pb, d_bank, beta, F, xi=xi_b)
        assert base is not None and pb.refl is not None
        result["refit"] = {"data": de["data"] - base[0], "total": e - base[3],
                           "prior_xi": pb.refl.log_density(xi_b) - pb.refl.log_density(xi)}
        log(f"  bank start with the field refitted: intensities {result['refit']['data']:+.0f}, field prior "
            f"{result['refit']['prior_xi']:+.0f}, total {result['refit']['total']:+.0f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
