#!/usr/bin/env python3
r"""Candidate radiometric laws scored on the per-bin table: held-out likelihood and agreement of the passes.

Usage: python scripts/residual_models.py TABLE.npz [--models base range-pow ...] [--out-dir figs]

The table (`scripts/residual_table.py`) holds, per scored bin, the unit intensity :math:`S_0` of the
current model without the reflectivity field and the mean range :math:`r`, beam angle :math:`\beta`
and incidence cosine :math:`\cos\varphi` of the bin's ground. A candidate changes the echo by a
factor evaluated at those means,

.. math::

    \bar I_b = g_c(r_b)\Bigl(\exp\bigl[u_{t,c} + \xi(q_b) + \Delta(r_b, \beta_b, \varphi_b)\bigr] S_{0,b} + N_c\Bigr),

with :math:`\Delta = 0` the current model, so the geometry (ground length per bin, shadows) stays that of
the render. :math:`\xi` lives on a mesh of 1 m with the prior of the match factor (Matérn,
:math:`\rho = 2` m, :math:`\sigma = 1`), :math:`u_{t,c}` has its Gaussian prior, and everything is fitted
by maximum a posteriori (L-BFGS) under the bin mixture of the ping factor.

Terms of :math:`\Delta`, each per side unless noted (`MODELS` names their combinations):

* range: :math:`a \log r + b r` (`pow`), or piecewise linear in :math:`r` (`spline`). The gain and the
  spreading and absorption exponents enter the echo together, so this is their joint error.
  `shared` replaces the two sides' gains by one law :math:`g(r) = r^a e^{b r}` for both, and `tvg30`
  sets it to :math:`r^3 e^{4 \alpha r}`, the gain that cancels spreading and absorption of a seabed echo.
* `floor`: the background :math:`g_c(r) N_c` replaced by :math:`P_t\, e^{f_0} (r / 15)^{f_1}`, a floor that
  follows the transmit power :math:`P_t`, as the water column does.
* beam: the Gaussian refitted (`gauss`); plus a second, wide Gaussian (`gauss2`);
  :math:`\operatorname{sinc}^2(W \sin(\beta - \beta_0))` on a floor (`sinc`); piecewise linear in
  :math:`\beta` (`spline`).
* angular law in place of :math:`\cos^2\varphi` (both sides): :math:`\cos\varphi`, Lommel--Seeliger
  (`ls`); :math:`\cos^n\varphi` (`pow`); :math:`\cos^n\varphi + a\, e^{-\tan^2\varphi / 2 s^2} / \cos^4\varphi`,
  diffuse plus a facet lobe about the normal (`mixed`); piecewise linear in :math:`\varphi` (`spline`).
* `slope`: a reflectivity of the ping-side that varies along range, :math:`v_{t,c} (r - 30) / 10`,
  :math:`v \sim \mathcal{N}(0, 0.5^2)`.
* `smooth`: the echo spread over neighbouring bins, :math:`S' = (1 - 2w) S_b + w (S_{b-1} + S_{b+1}) + \tau S_{b-1}`.
* `noise`: the shape :math:`L_c`, spike rate :math:`\epsilon_c` and level :math:`\eta_c` refitted.
* `proto`: the prototype's gain per side and constant background (`scripts/prototype_theta.py`) in
  place of the table's, for tables written with `--theta tvg`; `proto: "file"` also takes the
  beam and the linear gain of the state the table was written at (its :math:`\psi`).
* `ang: "plus"`: :math:`\cos^2\varphi + c`, a level the echo keeps at grazing incidence.
* `fit_pass`: the laws are fitted on that pass alone (0 is A, 1 is B) and kept for the scores, e.g. a
  beam calibrated on the pass over level ground.
* `temper`: the bin likelihood raised to this power in the fit (not in the scores), the weight
  neighbouring pings and bins would have if their correlation were counted; `L`: the shape set to this.

Scores, with blocks of 8 bins of each ping-side held out (20%):

1. the held-out log likelihood per bin with the shared field;
2. with the candidate's parameters kept and one field per pass, the standard deviation of
   :math:`\hat\xi_B - \hat\xi_A` over the vertices both passes cover;
3. transfer: the held-out likelihood of each pass on that overlap under the *other* pass's field,
   against no field (0) and its own field (1);
4. over 1 m ground cells both passes see, the residual
   :math:`e = \log(Y_B / \bar I_B) - \log(Y_A / \bar I_A)` without any field.

The last part splits :math:`e` (no field) into what laws of the viewing geometry remove, the noise
between odd and even pings of one pass, and the rest, and registers the two passes' residual
maps against each other by cross-correlation: a peak off zero is a geometric error (a depth
error :math:`\delta` moves the footprints of the two passes apart by
:math:`\delta\, h (1/x_A + 1/x_B)` for altitude :math:`h` and ground ranges :math:`x`).

Only the intensities and the navigation are used; the 3DSS bathymetry is never read.
"""
import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import sparse as jsparse
from jax.flatten_util import ravel_pytree
from matplotlib.colors import LinearSegmentedColormap
from prototype_theta import GAIN_DB as PROTO_GAIN_DB
from prototype_theta import NOISE as PROTO_NOISE
from residual_common import DB_TO_LOG_I, SIDES, Bins, cells, load
from scipy.optimize import minimize

from bbf import mesh as M
from bbf.ping.factor import log_gamma
from bbf.prior.spde import precision

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

RHO_XI, SIGMA_XI = 2.0, 1.0
R_REF = 30.0
R_FLOOR = 15.0
SLOPE_SD = 0.5
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
MODELS = {
    "base": {},
    "range-pow": {"range": "pow"},
    "range-spline": {"range": "spline"},
    "range-tvg30": {"range": "tvg30"},
    "range-shared": {"range": "shared"},
    "floor": {"floor": True},
    "range-shared+floor": {"range": "shared", "floor": True},
    "beam-gauss": {"beam": "gauss"},
    "beam-gauss2": {"beam": "gauss2"},
    "beam-sinc": {"beam": "sinc"},
    "beam-spline": {"beam": "spline"},
    "ang-ls": {"ang": "ls"},
    "ang-pow": {"ang": "pow"},
    "ang-mixed": {"ang": "mixed"},
    "ang-spline": {"ang": "spline"},
    "slope": {"slope": True},
    "smooth": {"smooth": True},
    "noise": {"noise": True},
    "range-pow+beam-gauss": {"range": "pow", "beam": "gauss"},
    "range-shared+beam-gauss": {"range": "shared", "beam": "gauss"},
    "range-shared+beam-gauss+ang-pow+floor": {"range": "shared", "beam": "gauss", "ang": "pow", "floor": True},
    "range-pow+beam-gauss+ang-pow": {"range": "pow", "beam": "gauss", "ang": "pow"},
    "range-pow+beam-gauss2+ang-mixed": {"range": "pow", "beam": "gauss2", "ang": "mixed"},
    "range-spline+beam-spline": {"range": "spline", "beam": "spline"},
    "range-spline+beam-spline+ang-spline": {"range": "spline", "beam": "spline", "ang": "spline"},
    "range-spline+beam-spline+slope": {"range": "spline", "beam": "spline", "slope": True},
    "range-spline+beam-spline+noise": {"range": "spline", "beam": "spline", "noise": True},
    "all": {"range": "spline", "beam": "spline", "ang": "spline", "slope": True, "smooth": True, "noise": True},
    # for tables written with --theta tvg: the prototype's laws on the same bins, and the candidates on the shared gain
    "proto": {"proto": True},
    "proto@psi": {"proto": "file"},
    "proto+range-pow+beam-gauss": {"proto": True, "range": "pow", "beam": "gauss"},
    "beam-gauss+noise": {"beam": "gauss", "noise": True},
    "beam-gauss+ang-pow": {"beam": "gauss", "ang": "pow"},
    "beam-gauss+ang-plus": {"beam": "gauss", "ang": "plus"},
    "beam-gauss+ang-spline": {"beam": "gauss", "ang": "spline"},
    "beam-spline+noise": {"beam": "spline", "noise": True},
    "beam-spline+ang-plus+noise": {"beam": "spline", "ang": "plus", "noise": True},
    "beam-gauss+temper0.5": {"beam": "gauss", "temper": 0.5},
    "beam-gauss+temper0.25": {"beam": "gauss", "temper": 0.25},
    "beam-gauss+L2.4": {"beam": "gauss", "L": 2.4},
    "beam-gauss+L1.2": {"beam": "gauss", "L": 1.2},
    "beam-gauss+ang-plus+noise": {"beam": "gauss", "ang": "plus", "noise": True},
    "beam-gauss@B": {"beam": "gauss", "fit_pass": 1},
    "beam-spline@B": {"beam": "spline", "fit_pass": 1},
    "beam-spline@A": {"beam": "spline", "fit_pass": 0},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Data:
    """The table's bins as arrays for the fits, with the field's mesh and the held-out mask."""

    def __init__(self, table: Path, h: float = 1.0, seed: int = 0):
        b = load(table)
        self.b = b
        z = np.load(table)
        S0g = z["f_S0"].astype(float)
        nb = S0g.shape[-1]
        self.rm = np.clip(S0g[b.ping, b.side, np.maximum(b.b - 1, 0)] / b.S0, 0.0, 50.0)
        self.rp = np.clip(S0g[b.ping, b.side, np.minimum(b.b + 1, nb - 1)] / b.S0, 0.0, 50.0)
        lo, hi = b.q.min(axis=0), b.q.max(axis=0)
        self.mesh = M.build(lo[0], hi[0], lo[1], hi[1], h=h, buffer=2 * h, origin=(0.0, 0.0))
        tri, bary, _ = (np.asarray(a) for a in M.locate(self.mesh, b.q))
        self.vidx = self.mesh.triangles[tri]
        self.bary = bary
        self.Lam = precision(self.mesh, RHO_XI, SIGMA_XI)
        rng = np.random.default_rng(seed)
        self.test = rng.random((b.par["shape"][0], 2, nb // 8 + 1))[b.ping, b.side, b.b // 8] < 0.2
        self.n_ps = 2 * b.par["shape"][0]
        par = b.par
        self.side_par = {k: np.asarray(par[k], float) for k in ("noise", "eps", "eta", "zeta", "ell", "beta0", "omega",
                                                                 "log_kappa_mean", "log_kappa_std")}
        self.L = float(par["L"])
        # vertices each pass covers: hat weight of its training bins
        w = np.zeros((2, self.mesh.n_vertices))
        for p in (0, 1):
            m = (b.pass_id == p) & ~self.test
            np.add.at(w[p], self.vidx[m].ravel(), self.bary[m].ravel())
        self.cover = w
        self.overlap = (w >= 5.0).all(axis=0)
        self.in_overlap = self.overlap[self.vidx[np.arange(b.n), np.argmax(self.bary, axis=1)]]
        self.knots = {"range": np.arange(math.floor(b.r.min() / 3) * 3.0, b.r.max() + 3.0, 3.0),
                      "beam": np.radians(np.arange(math.floor(np.degrees(b.beta.min()) / 4) * 4.0,
                                                   np.degrees(b.beta.max()) + 4.0, 4.0)),
                      "ang": np.radians(np.arange(0.0, 91.0, 5.0))}


def init_params(D: Data, spec: dict, field: str) -> dict:
    b, sp = D.b, D.side_par
    u = np.zeros(D.n_ps)
    u[b.ps] = b.u_hat
    p: dict = {"u": u}
    if field == "shared":
        p["xi"] = np.zeros(D.mesh.n_vertices)
    elif field == "pass":
        p["xi"] = np.zeros((2, D.mesh.n_vertices))
    kind = spec.get("range")
    if kind == "pow":
        p["range"] = np.zeros((2, 2))
    elif kind == "shared":
        p["range"] = np.array([3.0, 40 * float(b.par["alpha"])])
    elif kind == "spline":
        p["range"] = np.zeros((2, len(D.knots["range"])))
    kind = spec.get("beam")
    if kind == "gauss":
        p["beam"] = np.stack([sp["beta0"], np.log(sp["omega"])], axis=1)
    elif kind == "gauss2":
        p["beam"] = np.stack([sp["beta0"], np.log(sp["omega"]), np.full(2, -3.0), sp["beta0"], np.log(2 * sp["omega"])], axis=1)
    elif kind == "sinc":
        p["beam"] = np.stack([sp["beta0"], np.log(0.443 / np.sin(sp["omega"] / 2)), np.full(2, -4.0)], axis=1)
    elif kind == "spline":
        p["beam"] = np.zeros((2, len(D.knots["beam"])))
    kind = spec.get("ang")
    if kind == "pow":
        p["ang"] = np.array([2.0])
    elif kind == "mixed":
        p["ang"] = np.array([2.0, -2.0, math.log(0.2)])
    elif kind == "plus":
        p["ang"] = np.array([-4.0])
    elif kind == "spline":
        p["ang"] = np.zeros(len(D.knots["ang"]))
    if spec.get("slope"):
        p["slope"] = np.zeros(D.n_ps)
    if spec.get("smooth"):
        p["smooth"] = np.array([-4.0, -6.0])
    if spec.get("floor"):
        g15 = DB_TO_LOG_I * (b.par["gain_db"][:, 0] + b.par["gain_db"][:, 1] * R_FLOOR + b.par["gain_db"][:, 2] * math.log10(R_FLOOR))
        p["floor"] = np.stack([np.log(sp["noise"]) + g15, np.ones(2)], axis=1)
    if spec.get("noise"):
        p["noise"] = np.stack([np.full(2, math.log(spec.get("L", D.L))), np.log(sp["eps"] / (1 - sp["eps"])),
                               np.log(sp["eta"])], axis=1)
    return {k: jnp.asarray(v) for k, v in p.items()}


def _gauss(beta, b0, log_w):
    return -4 * math.log(2) * (beta - b0) ** 2 / jnp.exp(2 * log_w)


def make_model(D: Data, spec: dict, field: str, xi_fixed=None):
    r"""(per-bin log likelihood, negative log posterior) as functions of the parameters; `xi_fixed` (2, n) gives each pass a set field."""
    b, sp = D.b, D.side_par
    side, ps, pid = jnp.asarray(b.side), jnp.asarray(b.ps), jnp.asarray(b.pass_id)
    y, logS0, r, beta, log_g = (jnp.asarray(a) for a in (b.y, np.log(b.S0), b.r, b.beta, b.log_g))
    lc = jnp.log(jnp.clip(jnp.asarray(b.cos), 1e-3, 1.0))
    phi = jnp.arccos(jnp.clip(jnp.asarray(b.cos), 0.0, 1.0))
    tan2 = jnp.exp(-2 * lc) - 1.0
    vidx, bary = jnp.asarray(D.vidx), jnp.asarray(D.bary)
    rm, rp = jnp.asarray(D.rm), jnp.asarray(D.rp)
    c = {k: jnp.asarray(v)[side] for k, v in sp.items()}
    lB0 = _gauss(beta, c["beta0"], jnp.log(c["omega"]))
    r_bin, off_bin, alpha = jnp.asarray(b.par["r_bin"])[jnp.asarray(b.b)], jnp.asarray(b.offset), float(b.par["alpha"])
    spike = c["eta"] + c["zeta"] * jnp.exp(-r_bin / c["ell"])
    Lam = jsparse.BCOO.from_scipy_sparse(D.Lam.tocoo())
    off = np.zeros(D.n_ps)
    off[b.ps] = b.offset
    side_of_ps = np.tile([0, 1], D.n_ps // 2)
    m_u = jnp.asarray(sp["log_kappa_mean"][side_of_ps] + off)
    s_u = jnp.asarray(sp["log_kappa_std"][side_of_ps])
    kn = {k: jnp.asarray(v) for k, v in D.knots.items()}
    # background before gain, per bin: the table's law (constant unless written with --theta tvg)
    bg = c["noise"] * (r_bin / 20.0) ** float(b.par["noise_slope"]) * jnp.exp(float(b.par["noise_power"]) * off_bin)
    L_main = float(spec.get("L", D.L))
    temper = float(spec.get("temper", 1.0))
    proto = spec.get("proto")
    lB_state = lB0
    if proto:
        gdb = np.array(PROTO_GAIN_DB, float)
        psi_file = np.asarray(b.par["psi_file"], float)
        if proto == "file":
            if not np.isfinite(psi_file).all():
                raise ValueError("the table's state has no psi")
            gdb[:, 1] = psi_file[:, 2]
            # the state's beam in place of the table's
            lB_state = _gauss(beta, jnp.asarray(psi_file[:, 0])[side], jnp.log(jnp.asarray(psi_file[:, 1])[side]))
        gs = jnp.asarray(gdb)[side]
        log_g = DB_TO_LOG_I * (gs[:, 0] + gs[:, 1] * r_bin + gs[:, 2] * jnp.log10(r_bin))
        bg = jnp.asarray(PROTO_NOISE)[side] + 0.0 * r_bin
    xi_fix = None if xi_fixed is None else jnp.asarray(xi_fixed)

    def per_side(f, P):
        return jnp.where(side == 0, f(P[0]), f(P[1]))

    def field_at(p):
        if xi_fix is not None:
            return jnp.sum(xi_fix[pid[:, None], vidx] * bary, axis=1)
        if field == "shared":
            return jnp.sum(p["xi"][vidx] * bary, axis=1)
        if field == "pass":
            return jnp.sum(p["xi"][pid[:, None], vidx] * bary, axis=1)
        return 0.0

    def delta(p):
        d = lB_state - lB0
        kind = spec.get("range")
        if kind == "pow":
            d += per_side(lambda q: q[0] * jnp.log(r / R_REF) + q[1] * (r - R_REF) / 10.0, p["range"])
        elif kind == "spline":
            d += per_side(lambda q: jnp.interp(r, kn["range"], q), p["range"])
        elif kind == "tvg30":
            d += 3.0 * jnp.log(r / R_REF) + 4 * alpha * (r - R_REF) - log_g
        elif kind == "shared":
            d += p["range"][0] * jnp.log(r / R_REF) + p["range"][1] * (r - R_REF) / 10.0 - log_g
        kind = spec.get("beam")
        if kind == "gauss":
            d += per_side(lambda q: _gauss(beta, q[0], q[1]), p["beam"]) - lB0
        elif kind == "gauss2":
            d += per_side(lambda q: jnp.logaddexp(_gauss(beta, q[0], q[1]), q[2] + _gauss(beta, q[3], q[4])), p["beam"]) - lB0
        elif kind == "sinc":
            d += per_side(lambda q: jnp.log(jnp.sinc(jnp.exp(q[1]) * jnp.sin(beta - q[0])) ** 2 + jnp.exp(q[2])), p["beam"]) - lB0
        elif kind == "spline":
            d += per_side(lambda q: jnp.interp(beta, kn["beam"], q), p["beam"])
        kind = spec.get("ang")
        if kind == "ls":
            d += -lc
        elif kind == "pow":
            d += (p["ang"][0] - 2.0) * lc
        elif kind == "mixed":
            n, la, ls = p["ang"]
            d += jnp.logaddexp(n * lc, la - tan2 / (2 * jnp.exp(2 * ls)) - 4 * lc) - 2 * lc
        elif kind == "spline":
            d += jnp.interp(phi, kn["ang"], p["ang"])
        elif kind == "plus":
            d += jnp.logaddexp(2 * lc, p["ang"][0]) - 2 * lc
        if spec.get("slope"):
            d += p["slope"][ps] * (r - R_REF) / 10.0
        if spec.get("smooth"):
            w = jax.nn.sigmoid(p["smooth"][0]) / 3
            d += jnp.log((1 - 2 * w) + w * (rm + rp) + jnp.exp(p["smooth"][1]) * rm)
        return d

    def loglik(p):
        eta_sig = p["u"][ps] + field_at(p) + logS0 + delta(p)
        g = jnp.exp(log_g)
        if spec.get("floor"):
            q = p["floor"][side]
            mean = g * jnp.exp(eta_sig) + jnp.exp(q[:, 0] + q[:, 1] * jnp.log(r_bin / R_FLOOR) + off_bin)
        else:
            mean = g * (jnp.exp(eta_sig) + bg)
        if spec.get("noise"):
            q = p["noise"][side]
            L, eps, sp_mean = jnp.exp(q[:, 0]), jax.nn.sigmoid(q[:, 1]), mean + g * (jnp.exp(q[:, 2]) + spike - c["eta"])
        else:
            L, eps, sp_mean = L_main, c["eps"], mean + g * spike
        return jnp.logaddexp(jnp.log1p(-eps) + log_gamma(y, L, mean), jnp.log(eps) + log_gamma(y, 1.0, sp_mean)), mean

    def neg_log_post(p, mask):
        ll = temper * jnp.sum(jnp.where(mask, loglik(p)[0], 0.0))
        pr = 0.5 * jnp.sum(((p["u"] - m_u) / s_u) ** 2)
        if "xi" in p:
            x = jnp.atleast_2d(p["xi"])
            pr += 0.5 * sum(x[i] @ (Lam @ x[i]) for i in range(x.shape[0]))
        if "slope" in p:
            pr += 0.5 * jnp.sum((p["slope"] / SLOPE_SD) ** 2)
        for k in ("range", "beam", "ang"):
            if spec.get(k) == "spline":
                pr += 0.5 * 1e-2 * jnp.sum(p[k] ** 2) + 0.5 * 10.0 * jnp.sum(jnp.diff(p[k], n=2, axis=-1) ** 2)
        return -ll + pr

    return jax.jit(loglik), neg_log_post


def fit(D: Data, spec: dict, field: str, init: dict | None = None, hold: tuple = (), xi_fixed=None, mask=None,
        maxiter: int = 1500, ftol: float = 2e-8) -> tuple[dict, np.ndarray, np.ndarray]:
    """MAP fit on the training bins; returns (parameters, per-bin log likelihood, per-bin mean). `hold` names parameters kept at `init`."""
    t0 = time.time()
    loglik, nlp = make_model(D, spec, field, xi_fixed)
    p0 = init_params(D, spec, field)
    if init is not None:
        for k, v in init.items():
            if k in p0 and np.shape(v) == np.shape(p0[k]):
                p0[k] = jnp.asarray(v)
    fixed = {k: p0[k] for k in hold if k in p0}
    free = {k: v for k, v in p0.items() if k not in fixed}
    x0, unravel = ravel_pytree(free)
    mask = jnp.asarray(~D.test if mask is None else mask)
    f = jax.jit(jax.value_and_grad(lambda x: nlp({**fixed, **unravel(x)}, mask)))

    def fun(x):
        v, g = f(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    res = minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                                                               "maxcor": 30, "ftol": ftol, "gtol": 1e-6})
    log(f"    fit {field}{'' if xi_fixed is None else ' (fields given)'}: {res.nit} iterations, {res.nfev} evaluations, "
        f"-log posterior {res.fun:.0f}, {time.time() - t0:.0f}s")
    p = {**fixed, **unravel(jnp.asarray(res.x))}
    ll, mean = loglik(p)
    return {k: np.asarray(v) for k, v in p.items()}, np.asarray(ll), np.asarray(mean)


def cell_residual(D: Data, mean: np.ndarray, size: float = 1.0, min_bins: int = 10, sel=None):
    r"""Per-pass cell sums: (centres (m, 2), :math:`\log(Y / \bar I)` (2, m) with NaN below `min_bins`)."""
    b = D.b
    ci, centres = cells(b.q, size)
    out = np.full((2, len(centres)), np.nan)
    for p in (0, 1):
        m = b.pass_id == p
        if sel is not None:
            m &= sel
        cnt = np.bincount(ci[m], minlength=len(centres))
        Y = np.bincount(ci[m], b.y[m], minlength=len(centres))
        Ib = np.bincount(ci[m], mean[m], minlength=len(centres))
        ok = cnt >= min_bins
        out[p, ok] = np.log(Y[ok] / Ib[ok])
    return centres, out


def spread(e) -> tuple[float, float]:
    """(IQR, robust sd from the IQR) of the finite values."""
    e = e[np.isfinite(e)]
    iqr = float(np.subtract(*np.percentile(e, [75, 25])))
    return iqr, iqr / 1.349


def score(D: Data, name: str, spec: dict, init: dict | None, maxiter: int) -> tuple[dict, dict]:
    """Fit candidate `name` and compute its scores; returns (scores, shared-field parameters)."""
    b = D.b
    t0 = time.time()
    def with_laws(mask=None):
        """Joint fit, the laws first with u and the field held: from a fitted (u, field) the few law
        parameters are badly scaled against the rest, and the joint fit alone stops on its tolerance at once."""
        start = init
        if init is not None and any(k not in ("u", "xi", "slope") for k in init_params(D, spec, "shared")):
            start, _, _ = fit(D, spec, "shared", init=init, hold=("u", "xi", "slope"), mask=mask, maxiter=maxiter // 2, ftol=1e-11)
        return fit(D, spec, "shared", init=start, mask=mask, maxiter=maxiter, ftol=5e-9)

    if "fit_pass" in spec:
        # the laws from one pass alone (its training bins), then kept while u and the field are fitted to both
        pc, _, _ = with_laws(~D.test & (b.pass_id == spec["fit_pass"]))
        laws = tuple(k for k in pc if k not in ("u", "xi", "slope"))
        p, ll, mean = fit(D, spec, "shared", init={**(init or {}), **{k: pc[k] for k in laws}}, hold=laws, maxiter=maxiter)
    else:
        p, ll, mean = with_laws()
    out = {"model": name, "ll_test": float(ll[D.test].mean()), "ll_train": float(ll[~D.test].mean()),
           "xi_sd": float(p["xi"][D.cover.sum(axis=0) >= 5].std())}
    radiometric = tuple(k for k in p if k not in ("u", "xi", "slope"))
    # one field per pass, the candidate's laws kept
    pp, llp, _ = fit(D, spec, "pass", init={**p, "xi": np.stack([p["xi"], p["xi"]])}, hold=radiometric, maxiter=maxiter // 2)
    dxi = (pp["xi"][1] - pp["xi"][0])[D.overlap]
    out["dxi_sd"] = float(dxi.std())
    out["dxi_iqr_sd"] = spread(dxi)[1]
    out["ll_test_pass"] = float(llp[D.test].mean())
    # no field; transfer of each pass's field to the other
    p0, ll0, mean0 = fit(D, spec, "none", init=p, hold=radiometric, maxiter=maxiter // 2)
    _, llx, _ = fit(D, spec, "none", init=p, hold=radiometric, xi_fixed=pp["xi"][::-1], maxiter=maxiter // 2)
    sel = D.test & D.in_overlap
    for pi, pn in enumerate("AB"):
        m = sel & (b.pass_id == pi)
        own, none, other = llp[m].mean(), ll0[m].mean(), llx[m].mean()
        out[f"transfer_{pn}"] = float((other - none) / (own - none))
        out[f"ll_none_{pn}"], out[f"ll_own_{pn}"], out[f"ll_other_{pn}"] = float(none), float(own), float(other)
    _, rc = cell_residual(D, mean0)
    out["cell_e_iqr_nofield"], _ = spread(rc[1] - rc[0])
    _, rc = cell_residual(D, mean)
    out["cell_e_iqr_shared"], _ = spread(rc[1] - rc[0])
    out["ll_test_none"] = float(ll0[D.test].mean())
    for pi, pn in enumerate("AB"):
        for ci, cn in enumerate(SIDES):
            m = D.test & (b.pass_id == pi) & (b.side == ci)
            out[f"ll_test_{pn}_{cn}"] = float(ll[m].mean())
    out["time"] = time.time() - t0
    out["params"] = {k: np.round(np.asarray(v), 4).tolist() for k, v in p.items() if k in radiometric and np.size(v) <= 12}
    log(f"{name}: held-out ll/bin {out['ll_test']:+.4f} (no field {out['ll_test_none']:+.4f}, per pass {out['ll_test_pass']:+.4f}); "
        f"xi sd {out['xi_sd']:.2f}; d-xi sd {out['dxi_sd']:.2f}; transfer A {out['transfer_A']:+.2f} B {out['transfer_B']:+.2f}; "
        f"cell e IQR {out['cell_e_iqr_nofield']:.2f} / {out['cell_e_iqr_shared']:.2f}; {out['time']:.0f}s")
    return out, {**p, "_mean0": mean0, "_p0": p0}


def registration(D: Data, mean0: np.ndarray, out: Path, size: float = 0.5, max_shift: int = 24, tile: float = 30.0):
    r"""Cross-correlation of the passes' no-field residual maps against a shift of B; prints peaks, overall and per tile."""
    centres, rc = cell_residual(D, mean0, size=size, min_bins=4)
    ij = np.round(centres / size - 0.5).astype(int)
    ij -= ij.min(axis=0)
    shape = ij.max(axis=0) + 1
    maps = np.full((2, *shape), np.nan)
    for p in (0, 1):
        maps[p, ij[:, 0], ij[:, 1]] = rc[p]
    # high-pass: remove a 5 m running mean, so the registration uses texture and not the swath-wide trends
    from scipy.ndimage import uniform_filter
    hp = []
    for p in (0, 1):
        ok = np.isfinite(maps[p])
        w = uniform_filter(ok.astype(float), int(5 / size))
        sm = uniform_filter(np.where(ok, maps[p], 0.0), int(5 / size)) / np.maximum(w, 1e-9)
        hp.append(np.where(ok & (w > 0.3), maps[p] - sm, np.nan))
    A, B = hp

    def xcorr(sl_x, sl_y):
        res = np.full((2 * max_shift + 1, 2 * max_shift + 1), np.nan)
        a = A[sl_x, sl_y]
        for dx in range(-max_shift, max_shift + 1):
            for dy in range(-max_shift, max_shift + 1):
                bb = np.full(B.shape, np.nan)
                xs = slice(max(dx, 0), B.shape[0] + min(dx, 0))
                xo = slice(max(-dx, 0), B.shape[0] + min(-dx, 0))
                ys = slice(max(dy, 0), B.shape[1] + min(dy, 0))
                yo = slice(max(-dy, 0), B.shape[1] + min(-dy, 0))
                bb[xo, yo] = B[xs, ys]  # bb(x) = B(x + shift)
                bs = bb[sl_x, sl_y]
                m = np.isfinite(a) & np.isfinite(bs)
                if m.sum() > 300:
                    res[dx + max_shift, dy + max_shift] = np.corrcoef(a[m], bs[m])[0, 1]
        return res

    full = xcorr(slice(None), slice(None))
    pk = np.unravel_index(np.nanargmax(full), full.shape)
    log(f"registration of the residual maps ({size} m cells, 5 m high-pass): correlation at zero shift "
        f"{full[max_shift, max_shift]:+.3f}; peak {full[pk]:+.3f} at B shifted by "
        f"({(pk[0] - max_shift) * size:+.1f}, {(pk[1] - max_shift) * size:+.1f}) m (east, north)")
    nt = int(tile / size)
    tiles = []
    for i0 in range(0, shape[0] - nt // 2, nt):
        for j0 in range(0, shape[1] - nt // 2, nt):
            c = xcorr(slice(i0, i0 + nt), slice(j0, j0 + nt))
            if np.isfinite(c).sum() < c.size // 2:
                continue
            k = np.unravel_index(np.nanargmax(c), c.shape)
            tiles.append(((i0 + nt / 2) * size, (j0 + nt / 2) * size, (k[0] - max_shift) * size, (k[1] - max_shift) * size,
                          c[k], c[max_shift, max_shift]))
    for t in tiles:
        log(f"  tile at ({t[0]:5.0f}, {t[1]:5.0f}) m: peak {t[4]:+.2f} at shift ({t[2]:+.1f}, {t[3]:+.1f}) m; at zero {t[5]:+.2f}")
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ext = (-max_shift * size, max_shift * size, -max_shift * size, max_shift * size)
    im = ax[0].imshow(full.T, origin="lower", extent=ext, cmap=DIVERGING, vmin=-0.5, vmax=0.5)
    fig.colorbar(im, ax=ax[0], label="correlation")
    ax[0].set(xlabel="shift of B east (m)", ylabel="shift of B north (m)", title="Pass A against shifted pass B: residual texture")
    for a, mp, nm in zip(ax[1:], hp, "AB"):
        im = a.imshow(mp.T, origin="lower", cmap=DIVERGING, vmin=-1.5, vmax=1.5,
                      extent=(0, shape[0] * size, 0, shape[1] * size))
        fig.colorbar(im, ax=a, label="log residual, 5 m high-pass")
        a.set(xlabel="east (m)", ylabel="north (m)", title=f"Residual texture of pass {nm} (no field)")
    for t in tiles:
        ax[1].arrow(t[0], t[1], 4 * t[2], 4 * t[3], color="k", head_width=1.5, length_includes_head=True)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    log(f"wrote {out}")
    return full


def split_floor(D: Data, mean0: np.ndarray) -> float:
    r"""Variance of :math:`e` that the noise within a pass explains: from the residuals of odd against even pings, per cell."""
    b = D.b
    v = 0.0
    for p in (0, 1):
        halves = [cell_residual(D, mean0, sel=(b.pass_id == p) & (b.ping % 2 == h), min_bins=5)[1][p] for h in (0, 1)]
        dh = halves[1] - halves[0]
        v += spread(dh)[1] ** 2 / 4
    return v


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("table", type=Path)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--maxiter", type=int, default=600)
    args = ap.parse_args()
    D = Data(args.table)
    b: Bins = D.b
    log(f"{b.n} bins ({int(D.test.sum())} held out); field mesh {D.mesh.n_vertices} vertices, {int(D.overlap.sum())} covered by both passes; "
        f"{int(D.in_overlap.sum())} bins on the overlap")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, fits = [], {}
    base = None
    for name in args.models:
        res, p = score(D, name, MODELS[name], base, args.maxiter)
        if base is None:
            base = {k: v for k, v in p.items() if k in ("u", "xi")}
        results.append(res)
        fits[name] = p
        (args.out_dir / f"residual_models{args.tag}.json").write_text(json.dumps(results, indent=1))

    ref = results[0]
    print(f"\n{'model':38s} {'dLL/bin':>8s} {'dLL nofield':>11s} {'xi sd':>6s} {'dxi sd':>7s} {'transf A':>8s} {'transf B':>8s} {'e IQR 0':>8s} {'e IQR xi':>8s}")
    for r in results:
        print(f"{r['model']:38s} {r['ll_test'] - ref['ll_test']:+8.4f} {r['ll_test_none'] - ref['ll_test_none']:+11.4f} {r['xi_sd']:6.2f} "
              f"{r['dxi_sd']:7.2f} {r['transfer_A']:+8.2f} {r['transfer_B']:+8.2f} {r['cell_e_iqr_nofield']:8.2f} {r['cell_e_iqr_shared']:8.2f}")
    for r in results:
        if r["params"]:
            print(f"  {r['model']}: {r['params']}")

    # the passes' disagreement without a field: laws of the viewing geometry, noise, and the rest
    first, last = args.models[0], max(args.models, key=lambda n: fits[n]["_mean0"].size and
                                      next(r["ll_test_none"] for r in results if r["model"] == n))
    floor = split_floor(D, fits[last]["_mean0"])
    v = {}
    for n in (first, last):
        _, rc = cell_residual(D, fits[n]["_mean0"])
        v[n] = spread(rc[1] - rc[0])[1] ** 2
    print(f"\ncell residual e between the passes, no field (robust variance): {first} {v[first]:.3f}; {last} {v[last]:.3f}; "
          f"noise between odd and even pings {floor:.3f}")
    print(f"  shares of the {first} variance: viewing-geometry laws {(v[first] - v[last]) / v[first]:.1%}, "
          f"noise {floor / v[first]:.1%}, left (view-dependent, local) {(v[last] - floor) / v[first]:.1%}")
    registration(D, fits[last]["_mean0"], args.out_dir / f"residual_registration{args.tag}.png")

    # the fitted laws of the most flexible candidates against the current model
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    sp = D.side_par
    for name, ls in (("range-spline+beam-spline", "-"), ("range-pow+beam-gauss", "--"), ("beam-spline", ":"), ("range-spline", ":")):
        if name not in fits:
            continue
        p, spec = fits[name], MODELS[name]
        for c, colr in zip((0, 1), ("#2a78d6", "#eb6834")):
            m = b.side == c
            if spec.get("range") == "spline":
                kn = D.knots["range"]
                okk = (kn >= b.r[m].min()) & (kn <= b.r[m].max())
                ax[0].plot(kn[okk], p["range"][c][okk] - p["range"][c][okk].mean(), ls, color=colr, label=f"{name}, {SIDES[c]}")
            elif spec.get("range") == "pow":
                rr = np.linspace(b.r[m].min(), b.r[m].max(), 100)
                val = p["range"][c][0] * np.log(rr / R_REF) + p["range"][c][1] * (rr - R_REF) / 10
                ax[0].plot(rr, val - val.mean(), ls, color=colr, label=f"{name}, {SIDES[c]}")
            lo, hi = np.percentile(b.beta[m], [0.5, 99.5])
            bb = np.linspace(lo, hi, 200)
            lB0 = -4 * math.log(2) * (bb - sp["beta0"][c]) ** 2 / sp["omega"][c] ** 2
            if spec.get("beam") == "spline":
                val = lB0 + np.interp(bb, D.knots["beam"], p["beam"][c])
            elif spec.get("beam") == "gauss":
                val = -4 * math.log(2) * (bb - p["beam"][c][0]) ** 2 / np.exp(2 * p["beam"][c][1])
            else:
                continue
            ax[1].plot(np.degrees(bb), (val - val.max()) * 10 / math.log(10), ls, color=colr, label=f"{name}, {SIDES[c]}")
            if ls == "-":
                ax[1].plot(np.degrees(bb), (lB0 - lB0.max()) * 10 / math.log(10), "-", lw=0.8, color="k",
                           label="current Gaussian" if c == 0 else None)
    ax[0].set(xlabel="range (m)", ylabel="log factor on the echo (mean removed)", title="Range law: correction to gain x spreading x absorption")
    ax[1].set(xlabel="beam angle below the outboard axis (deg)", ylabel="dB", title="Beam pattern x angular response")
    for name, ls in (("ang-spline", "-"), ("ang-pow", "--"), ("ang-mixed", ":")):
        if name not in fits:
            continue
        ph = np.radians(np.linspace(1, 89, 200))
        lcc = np.log(np.cos(ph))
        pa = fits[name]["ang"]
        if name == "ang-spline":
            val = 2 * lcc + np.interp(ph, D.knots["ang"], pa)
        elif name == "ang-pow":
            val = pa[0] * lcc
        else:
            val = np.logaddexp(pa[0] * lcc, pa[1] - np.tan(ph) ** 2 / (2 * np.exp(2 * pa[2])) - 4 * lcc)
        ax[2].plot(np.degrees(ph), (val - val[0]) * 10 / math.log(10), ls, color="#2a78d6", label=name)
    ph = np.radians(np.linspace(1, 89, 200))
    ax[2].plot(np.degrees(ph), 20 * np.log10(np.cos(ph)), "-", lw=0.8, color="k", label=r"current $\cos^2\varphi$")
    ax[2].set(xlabel="incidence angle (deg)", ylabel="dB", title="Angular law (fitted alone)", ylim=(-45, 5))
    for a in ax:
        a.legend(frameon=False, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"residual_laws{args.tag}.png", dpi=100)
    log(f"wrote {args.out_dir / f'residual_laws{args.tag}.png'}")


if __name__ == "__main__":
    main()
