#!/usr/bin/env python3
r"""Compare saved inference states on the same passes: first echoes, profiles, predictive score, fields.

Usage: python scripts/compare_runs.py PATCHES.npz NAME=STATE.npz [NAME=STATE.npz ...] [--out figs/compare]
       [--score-every N] [--field-k K] [--no-fields]

Each state is a file `scripts/infer.py` wrote. Per state the script reports:

1. first-echo residuals :math:`r_0 - \tilde r` per pass and side (`bbf.ping.first_return`): the
   detected first echo against the model's nearest seabed range;
2. the predictive log density per bin of the scored pings, at the state's own level and noise
   model, over every bin beyond `MIN_RANGE` (one :math:`\log\kappa` per ping-side is fitted to the
   ping itself). The scored pings are those the state's run left out (`--hold-out`), else every
   `--score-every`-th: for a state fitted on all pings they are in-sample. Next to it the
   same bins scored by a plain gamma density about the predicted mean with the shape that suits
   the state best, which leaves the noise model out of the comparison, and the variance of
   :math:`\log(y / \bar I)`;
3. the standard deviation, over the vertices both passes see, of the difference between the
   log-reflectivity fields fitted to each pass alone at the state's depths (level `--field-k`);
4. cross-track profiles of the seabed through three pings of the first pass, with each side's
   first-echo range drawn as an arc about the sonar: the seabed should touch the arc and never
   come inside it.

A state without saved sonar parameters is taken to be the run `fr2` (prototype gain law, fitted
:math:`\psi`), whose level-3 parameters are those its log printed (`FR2`); one without a field (a
start) is only drawn and checked against the first echoes.

Only the intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tvg_theta as TVG
from prototype_theta import EPS, GAIN_DB, L_SHAPE, NOISE
from theta_io import theta_from

from bbf import mesh as M
from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import infer, jacobians, observed_vertices
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import load_depths, load_patch
from bbf.match import Reflectivity
from bbf.ping import (
    Theta,
    bin_ranges,
    log_gamma,
    log_mixture,
    mode,
    predicted,
    u_grid,
    unit_intensities,
    with_psi,
)
from bbf.ping.first_return import TAU, FirstReturn, _batched
from bbf.ping.geometry import side_frame

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

RHO, SIGMA, ALPHA = 5.0, 8.0, 3  # as scripts/infer.py
RHO_XI, SIGMA_XI = 2.0, 1.0
# run fr2 at its last level (k = 3), from runs/fr2.log, round 1
FR2 = {"k": 3, "eta": [9.17964302e07, 3.15917701e12], "zeta": [9.94163848e11, 1.64409627e09],
       "ell": [9.99788341, 1.79747671], "log_kappa_mean": [39.83462242, 38.92412911],
       "log_kappa_std": [0.65458147, 0.96512743]}
COLORS = ["#52514e", "#2a78d6", "#eb6834", "#1f9e89", "#8e5bd1"]
SIDES = ("port", "stbd")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclasses.dataclass
class State:
    name: str
    mesh: M.Mesh
    d: np.ndarray
    beta: np.ndarray
    xi: np.ndarray
    k: int               # level of `theta`
    theta: Theta         # the level's parameters, psi included
    base: Theta          # single-bin parameters with the run's gain, background and psi, to coarsen
    moves: np.ndarray    # (passes, 2) horizontal moves the run gave the passes
    hold_out: int
    min_range: float
    has_field: bool


def load_state(name: str, path: Path, n_passes: int) -> State:
    mesh, d, extra = load_depths(path)
    xi = np.asarray(extra["xi"], float) if "xi" in extra else np.zeros(mesh.n_vertices)
    beta = np.asarray(extra["beta"], float)
    if "theta_L" in extra:
        theta = theta_from(extra)
        k = int(extra["k"])
        base = Theta(noise=TVG.NOISE, L=L_SHAPE, eps=EPS, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE, ell=2.0,
                     log_kappa_mean=TVG.LOG_KAPPA_MEAN, log_kappa_std=2.0, gain_db=TVG.GAIN_DB,
                     noise_slope=TVG.NOISE_SLOPE, noise_power=TVG.NOISE_POWER, beta0=theta.beta0, omega=theta.omega)
        return State(name, mesh, d, beta, xi, k, theta, base, np.asarray(extra["moves"], float),
                     int(extra["hold_out"]), TVG.MIN_RANGE, True)
    base = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                 log_kappa_std=2.0, gain_db=GAIN_DB)
    if "psi" in extra:
        base = with_psi(base, np.asarray(extra["psi"], float))
    k = FR2["k"]
    theta = dataclasses.replace(coarsen_theta(base, k), **{f: np.array(v) for f, v in FR2.items() if f != "k"})
    return State(name, mesh, d, beta, xi, k, theta, base, np.zeros((n_passes, 2)), 0, 0.0, "xi" in extra)


def depth_at(st: State, q_world: np.ndarray) -> np.ndarray:
    """Depths below the reference height at world points (m, 2); NaN off the mesh."""
    tri, bary, inside = (np.asarray(a) for a in M.locate(st.mesh, st.mesh.to_local(q_world)))
    val = np.sum(st.d[st.mesh.triangles[tri]] * bary, axis=-1)
    return np.where(inside, val, np.nan)


def moved(passes, st: State):
    return [p.shifted(*mv) if np.any(mv) else p for p, mv in zip(passes, st.moves)]


def first_echo_residuals(st: State, passes, r0) -> list[np.ndarray]:
    r"""Per pass (n, 2): :math:`r_0 - \tilde r`, the first echo minus the model's nearest seabed range (m)."""
    fr = FirstReturn(r0)
    out = []
    for ps, r in zip(passes, r0):
        P = ps.local_poses(st.mesh.origin)
        rt = np.stack([np.asarray(_batched(st.mesh, c, TAU)(jnp.asarray(st.d), jnp.asarray(P), jnp.asarray(fr.u))[0])
                       for c in (0, 1)], axis=1)
        out.append(r - rt)
    return out


def score(st: State, passes, idx) -> dict:
    """Predictive log density per bin of pings idx (per pass) at the state's level; see the module docstring."""
    th = st.theta
    lls, n_bins, res, ys, means = [], 0, [], [], []
    for ps, sel in zip(passes, idx):
        cp = coarsen_pass(ps, st.k)
        u = u_grid(cp.n_bins, cp.dr, du=max(0.005, cp.dr / 3))
        poses = cp.local_poses(st.mesh.origin)[sel]
        S = np.asarray(unit_intensities(st.mesh, jnp.asarray(st.d), poses, th, u, cp.n_bins, cp.dr, jnp.asarray(st.xi)))
        r = np.asarray(bin_ranges(cp.n_bins, cp.dr))
        off = cp.log_power()[sel]
        for c in (0, 1):
            y = cp.y[sel, c]
            use = (r[None] >= TVG.MIN_RANGE) & (y > 0)

            def one(yy, SS, uu, oo, c=c, dr=cp.dr):
                u_hat, _ = mode(th, c, dr, yy, SS, uu, offset=oo, grid=49)
                I_bar, I_sp = predicted(SS, u_hat, th, c, dr, oo)
                return log_mixture(jnp.where(uu, yy, 1.0), I_bar, I_sp, th.L, th.at(c).eps), I_bar

            lp, I_bar = (np.asarray(a) for a in jax.jit(jax.vmap(one))(jnp.asarray(y), jnp.asarray(S[:, c]),
                                                                       jnp.asarray(use), jnp.asarray(off)))
            lls.append(lp[use].sum())
            n_bins += int(use.sum())
            ys.append(y[use])
            means.append(I_bar[use])
            res.append(np.log(y[use] / I_bar[use]))
    y, mean, res = np.concatenate(ys), np.concatenate(means), np.concatenate(res)
    shapes = np.exp(np.linspace(-3.0, 4.0, 141))
    by_shape = np.array([float(np.mean(np.asarray(log_gamma(y, sh, mean)))) for sh in shapes])
    best = int(np.argmax(by_shape))
    return {"ll_bin": float(np.sum(lls) / n_bins), "ll_bin_gamma": float(by_shape[best]), "gamma_shape": float(shapes[best]),
            "var_log_resid": float(np.var(res)), "bins": n_bins, "level": st.k}


def field_difference(st: State, passes, k: int, steps: int) -> dict:
    """sd of the difference between the fields fitted to each pass alone at the state's depths, over their overlap."""
    prior = PR.Prior(st.mesh, RHO, SIGMA, mu_beta=st.beta, Sigma_beta=np.array([[100.0]]), alpha=ALPHA)
    refl = Reflectivity(st.mesh, RHO_XI, SIGMA_XI)
    th = coarsen_theta(st.base, k)  # the shape of independent bins for every state alike
    fields, seen = [], []
    for ps in passes:
        cp = coarsen_pass(ps, k)
        u = u_grid(cp.n_bins, cp.dr, du=max(0.005, cp.dr / 3))
        fr = FirstReturn([I.first_echo(ps, st.base)])
        out = infer([cp], st.mesh, prior, th, st.d, st.beta, u, rounds=1, inner=0, calib_every=20, chunk=32, refl=refl,
                    first_return=fr, min_range=st.min_range, mode_grid=49, shadow_stop=False, log=lambda s: None)
        pb = out["problem"]
        _, _, _, xi, _ = gauss_newton(pb, st.d, st.beta, jacobians(pb, st.d, out["xi"]), xi=out["xi"], hold_depths=True,
                                      max_iter=steps, log=lambda s: None)
        fields.append(xi)
        seen.append(observed_vertices(pb, st.d))
    ov = seen[0] & seen[1]
    dxi = (fields[1] - fields[0])[ov]
    q = np.percentile(dxi, [25, 75])
    return {"dxi_sd": float(dxi.std()), "dxi_iqr_sd": float((q[1] - q[0]) / 1.349), "overlap_vertices": int(ov.sum()),
            "corr": float(np.corrcoef(fields[0][ov], fields[1][ov])[0, 1])}


def profiles(states, raw, r0, out: Path, reach: float = 45.0):
    """Cross-track profiles through three pings of the first pass, with the first-echo arcs."""
    ps = raw[0]
    picks = [ps.n_pings // 6, ps.n_pings // 2, 5 * ps.n_pings // 6]
    s = np.arange(-reach, reach + 0.01, 0.25)
    fig, axs = plt.subplots(len(picks), 1, figsize=(10, 3.4 * len(picks)), sharex=True)
    rows = []
    for ax, t in zip(axs, picks):
        ang = np.linspace(0, np.pi / 2, 90)
        for c, sign in ((0, -1.0), (1, 1.0)):
            if np.isfinite(r0[0][t, c]):
                ax.plot(sign * r0[0][t, c] * np.sin(ang), -r0[0][t, c] * np.cos(ang), color="#9a9892", lw=1.0, ls=":",
                        label="first-echo range (seabed touches, never inside)" if c == 0 else None)
        for i, st in enumerate(states):
            pose = moved(raw, st)[0].poses[t]
            _, _, _, _, o = (np.asarray(a) for a in side_frame(jnp.asarray(pose), 1))
            q = pose[None, :2] + s[:, None] * o[None, :2]
            z = st.mesh.z_ref - depth_at(st, q) - pose[2]  # height relative to the sonar
            ax.plot(s, z, color=COLORS[i % len(COLORS)], lw=2.0 if i else 1.4, ls="-" if i else "--", label=st.name)
            rows.append({"ping": int(t), "state": st.name,
                         **{f"z_at_{x:+d}": float(np.interp(x, s, z)) for x in (-30, -20, -10, 0, 10, 20, 30)}})
        ax.plot(0, 0, marker="v", color="#1f1e1c", ms=7)
        ax.set_ylabel("height from the sonar (m)")
        ax.set_title(f"pass {ps.name}, ping {t}: first echo port {r0[0][t, 0]:.1f} m, starboard {r0[0][t, 1]:.1f} m",
                     fontsize=10, loc="left")
        ax.grid(color="#e6e4df", lw=0.6)
        ax.set_ylim(-30, 2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    axs[-1].set_xlabel("across track (m): port negative, starboard positive")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("patches", type=Path)
    ap.add_argument("states", nargs="+", metavar="NAME=STATE.npz")
    ap.add_argument("--patches-used", nargs="+", default=["A", "B"], dest="names")
    ap.add_argument("--out", type=Path, default=Path("figs/compare"))
    ap.add_argument("--score-every", type=int, default=10)
    ap.add_argument("--field-k", type=int, default=27)
    ap.add_argument("--field-steps", type=int, default=6)
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()
    raw = [load_patch(args.patches, n) for n in args.names]
    states = [load_state(a.split("=", 1)[0], Path(a.split("=", 1)[1]), len(raw)) for a in args.states]
    tvg = Theta(noise=TVG.NOISE, L=L_SHAPE, eps=EPS, eta=10 * TVG.NOISE, zeta=100 * TVG.NOISE, ell=2.0,
                log_kappa_mean=TVG.LOG_KAPPA_MEAN, log_kappa_std=2.0, gain_db=TVG.GAIN_DB,
                noise_slope=TVG.NOISE_SLOPE, noise_power=TVG.NOISE_POWER)
    r0 = [I.first_echo(p, tvg) for p in raw]
    hold = max([st.hold_out for st in states] + [0]) or args.score_every
    idx = [np.flatnonzero(np.arange(p.n_pings) % hold == hold // 2) for p in raw]
    log(f"{len(states)} states; scoring every {hold}-th ping ({sum(len(i) for i in idx)} pings)")
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict = {"profiles": profiles(states, raw, r0, args.out / "profiles.png")}
    log(f"wrote {args.out / 'profiles.png'}")
    for st in states:
        passes = moved(raw, st)
        rec: dict = {"moves": st.moves.tolist(), "hold_out": st.hold_out}
        res = first_echo_residuals(st, passes, r0)
        for ps, e in zip(passes, res):
            for c, side in enumerate(SIDES):
                x = e[:, c][np.isfinite(e[:, c])]
                rec[f"first_echo_{ps.name}_{side}"] = {"median": float(np.median(x)), "within_0.3": float(np.mean(np.abs(x) < 0.3)),
                                                       "mad": float(np.median(np.abs(x - np.median(x))))}
                log(f"{st.name}: first-echo residual {ps.name} {side}: median {np.median(x):+.2f} m, |res| < 0.3 m on "
                    f"{np.mean(np.abs(x) < 0.3):.0%}")
        if not args.no_score and st.has_field:
            rec["score"] = score(st, passes, idx)
            sc = rec["score"]
            log(f"{st.name}: log density per bin {sc['ll_bin']:+.4f} at k = {sc['level']} over {sc['bins']} bins "
                f"({'held out' if st.hold_out else 'in sample'}); gamma about the mean {sc['ll_bin_gamma']:+.4f} "
                f"(shape {sc['gamma_shape']:.2f}); var log(y / mean) {sc['var_log_resid']:.2f}")
        if not args.no_fields and st.has_field:
            rec["fields"] = field_difference(st, passes, args.field_k, args.field_steps)
            f = rec["fields"]
            log(f"{st.name}: per-pass fields differ by sd {f['dxi_sd']:.2f} (IQR sd {f['dxi_iqr_sd']:.2f}; correlation "
                f"{f['corr']:+.2f}) over {f['overlap_vertices']} shared vertices")
        result[st.name] = rec
    (args.out / "compare.json").write_text(json.dumps(result, indent=1))
    log(f"wrote {args.out / 'compare.json'}")


if __name__ == "__main__":
    main()
