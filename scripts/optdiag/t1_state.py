r"""Diagnostics of the posterior's geometry at one state: what drives psi, how well it is identified, what the step does.

Usage: run.sh t1_state.py K STATE [EVERY]

1. Gradient in :math:`\psi` by term, pass, side and range band (scores of the bins times the
   derivative of :math:`\log \bar I`), checked against the exact gradient.
2. Curvature of :math:`\psi` in the GGN as other variable groups are eliminated (Schur complements),
   in units of its prior precision.
3. The Gauss--Newton step at :math:`\lambda = 10^{-3}`: its size before the limits, the predicted
   and the exact gain of the step as applied, of its parts alone, and by term.
4. Per vertex: the step, gradient and GGN diagonal by factor, saved for `t1_plot.py`.
"""
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from common import OUT, PSI_SD, log, setup
from gn import predicted_gain
from sksparse.cholmod import cho_factor

from bbf.inference import ggn
from bbf.inference.gauss_newton import (
    D_MAX_STEP,
    PSI_HI,
    PSI_LO,
    PSI_MAX_STEP,
    XI_MAX_STEP,
)
from bbf.inference.objective import data_term, render, value_and_grad
from bbf.ping.factor import (
    DB_TO_LOG_I,
    bin_ranges,
    laplace_ping_side,
    log_gamma,
    predicted,
)
from bbf.ping.first_return import first_return_term
from bbf.ping.shadow import log_shadow_side
from bbf.ping.theta import with_psi

k, state = int(sys.argv[1]), sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
s = setup(k, state, every)
pb = s.problem
lay = ggn.layout(pb)
n, ix, ib, nb, ip, iu = lay["n"], lay["xi"], lay["beta"], lay["n_beta"], lay["psi"], lay["u"]
names = ["port centre", "port width", "port g1", "stbd centre", "stbd width", "stbd g1"]
unit = np.array([np.degrees(1), np.degrees(1), 1.0] * 2)  # radians to degrees for printing
res: dict = {"k": k, "state": state}


def side_values(problem, S, psi, u_hat):
    """Per pass (n, 2): the ping factor and the shadow factor of every ping-side."""
    out = []
    th = with_psi(problem.theta, jnp.asarray(psi))
    for p, Sp in enumerate(S):
        vals = []
        for side in (0, 1):
            def f(y, SS, use, lab, val, u0, off, side=side):
                ping, uh = laplace_ping_side(th, side, problem.dr, y, SS, use, u0=u0, offset=off,
                                             temper=problem.temper)
                return ping, problem.temper * log_shadow_side(th, side, lab, SS, uh, val)
            a, b = jax.jit(jax.vmap(f))(jnp.asarray(problem.y[p][:, side]), jnp.asarray(Sp[:, side]),
                                        jnp.asarray(problem.uses()[p][:, side]),
                                        jnp.asarray(problem.labels[p][:, side]),
                                        jnp.asarray(s.valid[p][:, side]), jnp.asarray(u_hat[p][:, side]),
                                        jnp.asarray(s.offsets[p]))
            vals.append(np.stack([np.asarray(a), np.asarray(b)], axis=-1))
        out.append(np.stack(vals, axis=1))  # (n, 2, 2): ping-side, (ping, shadow)
    return out


def terms(d, beta, psi, xi, u0):
    """Every term of the log posterior at a point, and the data term by pass and side."""
    S = render(pb, d, psi, xi)
    _, u_hat = data_term(pb)(*[[jnp.asarray(x) for x in S], None, jnp.asarray(psi)])
    sv = side_values(pb, S, psi, [np.asarray(u) for u in u_hat])
    out = {"prior_d": pb.prior.log_density(d, beta), "prior_xi": pb.xi_log_prior(xi)[0],
           "prior_psi": pb.psi_log_prior(psi)[0],
           "first_return": first_return_term(pb.mesh, d, pb.poses, s.first_return)[0]}
    for p, nm in enumerate("AB"):
        for c, sd in enumerate(("port", "stbd")):
            out[f"ping {nm} {sd}"] = float(sv[p][:, c, 0].sum())
            out[f"shadow {nm} {sd}"] = float(sv[p][:, c, 1].sum())
    out["total"] = float(sum(out.values()))
    return out


F = data_term(pb)
ell, (gd, gb), det = value_and_grad(pb, s.d, s.beta, F, psi=s.psi, xi=s.xi)
log(f"k={k} {state}: ell {ell:.1f}, data {det['data']:.1f}, first return {det['first_return_value']:.1f}")
J, Jb = s.jac()

# 1. gradient in psi by bins -------------------------------------------------------------------------
th = with_psi(pb.theta, s.psi)
r_bin = np.asarray(bin_ranges(pb.n_bins, pb.dr))
kg = DB_TO_LOG_I * r_bin
bands = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 60)]
rows = []
tot = np.zeros(6)
for p, nm in enumerate("AB"):
    for c, sd in enumerate(("port", "stbd")):
        st = th.at(c)
        Sp, up, y = det["S"][p][:, c], det["u"][p][:, c, None], pb.y[p][:, c]
        use = pb.uses()[p][:, c] & (y > 0)
        Ib, Isp = (np.asarray(x) for x in predicted(Sp, up, th, c, pb.dr))
        ys = np.where(y > 0, y, 1.0)
        lm = np.log1p(-st.eps) + np.asarray(log_gamma(ys, th.L, Ib))
        ls = np.log(st.eps) + np.asarray(log_gamma(ys, 1.0, Isp))
        resp = np.exp(lm - np.logaddexp(lm, ls))
        s_main = np.where(use, pb.temper * resp * th.L * (ys / Ib - 1), 0.0)
        s_spk = np.where(use, pb.temper * (1 - resp) * (ys / Isp - 1), 0.0)
        s_logI = s_main + s_spk * Ib / Isp  # score in log I-bar through S
        s_gain = s_main + s_spk  # score in log g
        a = np.exp(up) / (np.exp(up) * Sp + st.noise)
        gb0 = s_logI * a * Jb[p][:, c, :, 0]
        gw = s_logI * a * Jb[p][:, c, :, 1]
        gg = s_gain * kg[None, :]
        for lo, hi in bands:
            m = (r_bin >= lo) & (r_bin < hi)
            rows.append((nm, sd, lo, hi, int(use[:, m].sum()), gb0[:, m].sum(), gw[:, m].sum(), gg[:, m].sum(),
                         float(np.sum(s_main[:, m]) / max(np.sum(use[:, m] * resp[:, m]) * th.L * pb.temper, 1))))
        tot[3 * c: 3 * c + 3] += [gb0.sum(), gw.sum(), gg.sum()]
g_psi = det["grad_psi"].ravel()
g_psi_prior = pb.psi_log_prior(s.psi)[1].ravel()
log("gradient in psi (per radian, radian, dB/m): exact total, of which prior, ping bins (from scores), rest (shadow, Laplace)")
for j, nm in enumerate(names):
    log(f"  {nm:12s} total {g_psi[j]:+12.1f}  prior {g_psi_prior[j]:+10.1f}  ping bins {tot[j]:+12.1f}  "
        f"rest {g_psi[j] - g_psi_prior[j] - tot[j]:+12.1f}")
log("ping-bin gradient by pass, side and range band: bins, d/d centre, d/d width, d/d g1, mean (y/I - 1) of main bins")
for r in rows:
    log(f"  {r[0]} {r[1]:4s} {r[2]:2d}-{r[3]:2d} m  {r[4]:7d}  {r[5]:+11.1f} {r[6]:+11.1f} {r[7]:+11.1f}   {r[8]:+.3f}")
res["psi_grad_rows"], res["g_psi"], res["g_psi_prior"], res["g_psi_bins"] = rows, g_psi, g_psi_prior, tot

# 2. curvature of psi as groups are eliminated ------------------------------------------------------------
H = ggn.assemble(pb, J, det["S"], det["u"], Jb=Jb, psi=s.psi, fr=det["first_return"])
g = np.zeros(lay["size"])
g[:n], g[ib:ib + nb], g[ix:ix + n], g[ip:ip + 6] = gd, gb, det["grad_xi"], g_psi
grp = {"d": np.r_[0:n, ib:ib + nb], "xi": np.arange(ix, ix + n), "u": np.arange(iu, ip)}
Hcsr = sp.csr_matrix(H)
Hpp = np.asarray(Hcsr[ip:ip + 6][:, ip:ip + 6].todense())
sd_prior = PSI_SD.ravel()


def schur(elim):
    if not elim:
        return Hpp
    idx = np.sort(np.concatenate([grp[e] for e in elim]))
    Hr = sp.csc_matrix(Hcsr[idx][:, idx])
    C = np.asarray(Hcsr[idx][:, ip:ip + 6].todense())
    return Hpp - C.T @ cho_factor(Hr).solve(C)


log("psi in the GGN as groups are eliminated: marginal sd (deg, deg, dB/m), and the Newton step g/S of psi alone")
res["schur"] = {}
for elim in ([], ["u"], ["u", "xi"], ["u", "d"], ["u", "xi", "d"]):
    Sc = schur(elim)
    sd = np.sqrt(np.diag(np.linalg.inv(Sc)))
    ev = np.linalg.eigvalsh(Sc * np.outer(sd_prior, sd_prior))
    res["schur"]["+".join(elim) or "none"] = Sc
    log(f"  eliminated {'+'.join(elim) or 'none':8s}: sd {np.round(sd * unit, 3)}; eigenvalues in prior units "
        f"{np.array2string(ev, precision=1)}")
log(f"  prior sd {np.round(sd_prior * unit, 2)}")
Sfull = res["schur"]["u+xi+d"]
w, V = np.linalg.eigh(Sfull * np.outer(sd_prior, sd_prior))
log("  weakest directions of psi after eliminating everything (components in prior sd units):")
for j in range(3):
    log(f"    eigenvalue {w[j]:8.1f}: {np.round(V[:, j], 2)}")

# 3. the step -----------------------------------------------------------------------------------------------
lam = 1e-3
damp = lam * np.maximum(H.diagonal(), 1e-12)
damp[ip:] = 0.0
f = cho_factor(sp.csc_matrix(H + sp.diags(damp)))
raw = f.solve(g)
over = max(1.0, float(np.max(np.abs(raw[ip:ip + 6].reshape(2, 3)) / PSI_MAX_STEP)))
delta = raw / over
delta[:n] = np.clip(delta[:n], -D_MAX_STEP, D_MAX_STEP)
delta[ix:ix + n] = np.clip(delta[ix:ix + n], -XI_MAX_STEP, XI_MAX_STEP)
delta[ip:ip + 6] = (np.clip(s.psi + delta[ip:ip + 6].reshape(2, 3), PSI_LO, PSI_HI) - s.psi).ravel()
log(f"step at lambda {lam:.0e}: raw psi step {np.round(raw[ip:ip + 6] * unit, 2)} -> whole step shortened {over:.1f}x")
log(f"  raw depth step: max {np.abs(raw[:n]).max():.1f} m, {np.sum(np.abs(raw[:n]) > 1)} vertices > 1 m, "
    f"{np.sum(np.abs(raw[:n]) > 10)} > 10 m; after shortening {np.sum(np.abs(raw[:n]) / over > 1)} clipped; "
    f"raw xi step max {np.abs(raw[ix:ix + n]).max():.1f}, {np.sum(np.abs(raw[ix:ix + n]) / over > XI_MAX_STEP)} clipped")
res.update({"raw": raw[:ip + 6][np.r_[0:2 * n, ip:ip + 6]], "over": over})


def part(which):
    z = np.zeros_like(delta)
    for wname in which:
        if wname == "d":
            z[:n], z[ib:ib + nb] = delta[:n], delta[ib:ib + nb]
        elif wname == "xi":
            z[ix:ix + n] = delta[ix:ix + n]
        else:
            z[ip:ip + 6] = delta[ip:ip + 6]
    return z


base = terms(s.d, s.beta, s.psi, s.xi, det["u"])
log(f"  terms at the state: total {base['total']:.1f} (value_and_grad: {ell:.1f})")
res["terms0"], res["probes"] = base, {}
log("  exact vs predicted gain of the applied step and of its parts alone; then the exact gain by term")
for which in (["d", "xi", "psi"], ["d"], ["xi"], ["psi"], ["d", "xi"], ["xi", "psi"]):
    z = part(which)
    pred = predicted_gain(H, g, z, lay)
    t = terms(s.d + z[:n], s.beta + z[ib:ib + nb], s.psi + z[ip:ip + 6].reshape(2, 3), s.xi + z[ix:ix + n], det["u"])
    diff = {key: t[key] - base[key] for key in t}
    res["probes"]["+".join(which)] = (pred, diff)
    log(f"  {'+'.join(which):9s}: actual {diff['total']:+11.1f}  predicted {pred:+11.1f}  ratio {diff['total'] / pred:5.2f}")
    log("      " + ", ".join(f"{key} {v:+.0f}" for key, v in diff.items() if key != "total"))
for tscale in (0.25, 0.5, 2.0):
    z = part(["d", "xi", "psi"]) * tscale
    pred = predicted_gain(H, g, z, lay)
    t = terms(s.d + z[:n], s.beta + z[ib:ib + nb], s.psi + z[ip:ip + 6].reshape(2, 3), s.xi + z[ix:ix + n], det["u"])
    log(f"  applied step x {tscale}: actual {t['total'] - base['total']:+11.1f}  predicted {pred:+11.1f}")
    res["probes"][f"x{tscale}"] = (pred, t["total"] - base["total"])

# 4. per vertex ---------------------------------------------------------------------------------------------
_, g_fr, A, W = first_return_term(pb.mesh, s.d, pb.poses, s.first_return)
diag_fr = np.asarray(A.multiply(A).T @ W).ravel()
Pr = pb.prior.joint_precision().tocsr()
diag_prior = Pr.diagonal()[:n]
gd_prior = pb.prior.grad(s.d, s.beta)[0]
diag_all = H.diagonal()
res.update({"d": s.d, "xi": s.xi, "delta_d": delta[:n], "raw_d": raw[:n], "raw_xi": raw[ix:ix + n],
            "g_data": det["grad_d_data"], "g_fr": g_fr, "g_prior": gd_prior, "g_xi": det["grad_xi"],
            "diag_d": diag_all[:n], "diag_fr": diag_fr, "diag_prior": diag_prior,
            "diag_xi": diag_all[ix:ix + n], "diag_xi_prior": s.refl.Lambda.diagonal(),
            "poses": pb.poses, "vertices": pb.mesh.vertices})
(OUT / f"t1_k{k}_{state}_e{every}.pkl").write_bytes(pickle.dumps(res))
log("saved")
