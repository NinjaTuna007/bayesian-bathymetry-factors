r"""Shared setup of the optimiser diagnostics: the problem of one level as `scripts/infer.py` builds it.

Uses only the intensities and the navigation; the 3DSS bathymetry is never read.
"""
import dataclasses
import pickle
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from bbf import prior as PR
from bbf.inference import init as I
from bbf.inference.calibrate import calibrate
from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import jacobians, labels_of
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.inference.objective import Problem, beam_jacobian, render
from bbf.io import load_depths, load_patch
from bbf.match import Reflectivity
from bbf.ping import Theta, psi_of, u_grid, with_psi
from bbf.ping.first_return import FirstReturn
from bbf.ping.shadow import onset_valid, ping_use, water_column_lit_rate

CACHE = Path("/home/ola/rss27_sss_bathymetry/cache/rosbag2_2026_07_10-17_39_53_patches.npz")
RUNS = Path("/home/ola/rss27_sss_bathymetry/src/runs")
OUT = Path(__file__).resolve().parents[2] / "runs" / "optdiag"
FIGS = Path(__file__).resolve().parents[2] / "figs" / "optdiag"
GAIN_DB = np.array([[0.0, 0.65, 0.0], [0.0, 0.0, 4.1]])
L_SHAPE, NOISE, EPS = 0.48, np.array([9.33e8, 5.28e7]), np.array([0.055, 0.021])
H_MESH, BUFFER = 0.5, 10.0
RHO, SIGMA, ALPHA = 5.0, 8.0, 3
SIGMA_BETA = np.array([[100.0]])
RHO_XI, SIGMA_XI = 2.0, 1.0
PSI_SD = np.array([[np.radians(10.0), np.radians(15.0), 0.5]] * 2)
PSI_PROTO = np.stack([np.radians([50.0, 50.0]), np.radians([60.0, 60.0]), GAIN_DB[:, 1]], axis=1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclasses.dataclass
class Setup:
    problem: Problem
    d: np.ndarray
    beta: np.ndarray
    psi: np.ndarray
    xi: np.ndarray
    passes: list
    r0: list
    first_return: FirstReturn  # the problem's optional parts, typed as present
    refl: Reflectivity
    valid: list
    offsets: list
    J: list | None = None
    Jb: list | None = None

    def jac(self) -> tuple[list, list]:
        if self.J is None or self.Jb is None:
            self.J = jacobians(self.problem, self.d, self.xi)
            self.Jb = beam_jacobian(self.problem, self.d, xi=self.xi)
        return self.J, self.Jb


def setup(k: int = 27, state: str = "start", every: int = 1, temper: float = 1.0, tag: str = "",
          psi_sd=PSI_SD, sigma_xi: float = SIGMA_XI, fit_xi: bool = True) -> Setup:
    r"""The level-k problem at a saved state of run fr2 (`start`, `k27`, `k9`), built as `infer` builds its round.

    every: keep every this-many-th ping (1: all). At `start` the reflectivity is fitted with the depths
    held, as `infer` does in its first round; the fit is cached under `runs/optdiag` (per k, state,
    every and tag: give a tag when temper or sigma_xi differ, since the fit depends on them).
    """
    passes = [load_patch(CACHE, n) for n in ("A", "B")]
    theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
                  log_kappa_std=2.0, gain_db=GAIN_DB)
    mesh = I.mesh_for_passes(passes, h=H_MESH, buffer=BUFFER)
    r0_full = [I.first_echo(p, theta) for p in passes]
    if every > 1:
        passes = [dataclasses.replace(p, poses=p.poses[::every], y=p.y[::every], t=p.t[::every],
                                      power=None if p.power is None else p.power[::every]) for p in passes]
        r0_full = [r[::every] for r in r0_full]
    first_return = FirstReturn(r0_full)
    _, d, extra = load_depths(RUNS / f"fr2_{state}.npz")
    beta = extra["beta"]
    psi = extra["psi"] if "psi" in extra else psi_of(theta)
    xi = extra.get("xi")
    theta = with_psi(theta, psi)
    beta_start = load_depths(RUNS / "fr2_start.npz")[2]["beta"]
    prior = PR.Prior(mesh, RHO, SIGMA, mu_beta=beta_start, Sigma_beta=SIGMA_BETA, alpha=ALPHA)
    refl = Reflectivity(mesh, RHO_XI, sigma_xi)
    cp = [coarsen_pass(p, k) for p in passes]
    th = coarsen_theta(theta, k)
    u = u_grid(cp[0].n_bins, cp[0].dr, du=max(0.005, cp[0].dr / 3))
    poses = [p.local_poses(mesh.origin) for p in cp]
    ys = [p.y for p in cp]
    dr = cp[0].dr
    block_dr = th.block * dr
    labels = [labels_of(p, th) for p in cp]
    r0 = [I.first_echo(p, th) for p in cp]
    valid = [onset_valid(r, lab.shape[-1], block_dr, 1.5) for r, lab in zip(r0, labels)]
    th = dataclasses.replace(th, wc_lit=water_column_lit_rate(labels, r0, block_dr, 1.5))
    offsets = [p.log_power() for p in cp]
    base = Problem(mesh=mesh, prior=prior, theta=th, poses=poses, y=ys, labels=labels, u=u, dr=dr, chunk=32,
                   valid=valid, refl=refl, kappa_offset=offsets, first_return=first_return, temper=temper)
    cache = OUT / f"setup_k{k}_{state}_e{every}{tag}.pkl"
    OUT.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        c = pickle.loads(cache.read_bytes())
        th = dataclasses.replace(th, **c["theta"])
        xi = c["xi"]
    else:
        S = render(base, d, xi=xi)
        sub = [np.arange(0, len(p), max(50 // every, 1)) for p in poses]
        y_c = np.concatenate([y[s] for y, s in zip(ys, sub)])
        S_c = np.concatenate([Sp[s] for Sp, s in zip(S, sub)])
        use_c = np.concatenate([np.stack([ping_use(lab[t], ys[0].shape[-1], th.block, valid=val[t]) for t in s])
                                for lab, val, s in zip(labels, valid, sub)])
        th, _ = calibrate(y_c, S_c, use_c, th, dr, offset=np.concatenate([o[s] for o, s in zip(offsets, sub)]))
        log(f"theta fit: eta {np.asarray(th.eta)}, zeta {np.asarray(th.zeta)}, ell {np.asarray(th.ell)}, "
            f"log kappa {np.asarray(th.log_kappa_mean)} ± {np.asarray(th.log_kappa_std)}")
        if xi is None:
            xi = np.zeros(mesh.n_vertices)
            if fit_xi:
                fixed = dataclasses.replace(base, theta=th)
                _, _, _, xi, _ = gauss_newton(fixed, d, beta, jacobians(fixed, d, xi), xi=xi, hold_depths=True,
                                              max_iter=6, log=log)
        cache.write_bytes(pickle.dumps({"theta": {f: getattr(th, f) for f in
                                                  ("eta", "zeta", "ell", "log_kappa_mean", "log_kappa_std")},
                                        "xi": xi}))
    # the prior of psi stays at the prototype's values, as in `scripts/infer.py`
    problem = dataclasses.replace(base, theta=th, psi_prior=(PSI_PROTO, psi_sd))
    return Setup(problem, np.asarray(d, float), np.asarray(beta, float), np.asarray(psi, float),
                 np.asarray(xi, float), cp, r0_full, first_return, refl, valid, offsets)
