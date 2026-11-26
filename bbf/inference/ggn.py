r"""Generalised Gauss--Newton curvature of the log posterior as a sparse augmented matrix.

Variables, in order: depths :math:`d` (n), the log-reflectivity field :math:`\xi` (n, with
`Problem.refl`), trend :math:`\beta` (3, or 1 for a constant mean), and one :math:`u = \log\kappa`
per ping-side (all passes, ping-major, port then starboard). Keeping :math:`u` as variables (instead of eliminating them)
gives the same Gauss--Newton step and Laplace marginal for :math:`d`, while :math:`H` keeps
:math:`d`--:math:`d` entries only between vertices that share a bin.

Each data term contributes :math:`G^\top I G`, with :math:`I` the Fisher information of its
likelihood in a scalar prediction and :math:`G` that prediction's derivative:

* ping bin :math:`b` (bins the ping factor scores): prediction :math:`\log\bar I_b`, Fisher
  :math:`L\,r_b` with :math:`r_b` the main component's responsibility,
  :math:`\partial\log\bar I_b/\partial S_b = e^u / (e^u S_b + N)`,
  :math:`\partial\log\bar I_b/\partial u = e^u S_b / (e^u S_b + N)`;
* dark shadow block :math:`j` (lit blocks are left to the ping factor): prediction :math:`p_j`,
  weight :math:`1/(1 - p_j)^2`, the curvature of :math:`-\log(1 - p_j)` in :math:`p_j`, through the
  block mean of :math:`S` with :math:`\hat\kappa` held fixed; or, with the horizon factor
  (`bbf.ping.horizon`), its rows on the depths in place of these;
* the first-return and horizon factors: their own rows :math:`A` and weights :math:`W` on the depths;
* the ratio match (`Problem.ratio`, `bbf.match.ratio`): per cell, prediction :math:`\eta_v = \log \bar I_v`
  of each visit, Fisher :math:`\tau (\mathrm{diag}(k) - k k^\top / K)` with the held shapes :math:`k`, as rows
  :math:`A` and :math:`B` (`bbf.match.ratio.fisher_rows`) through :math:`\partial \bar I_b / \partial S_b = g_c(r_b) e^{u}`
  over all passes together. The gains :math:`u` it sees are held for the round
  (`objective.ratio_gains`), not the ping factor's variables, so it has no :math:`u` entries and
  :math:`H_{uu}` stays diagonal (`gauss_newton.predicted_gain` relies on that);

plus the prior's exact precision on :math:`(d, \beta)`, that of :math:`\xi`, and the Gaussian prior
on each :math:`u`.

:math:`S` reaches :math:`d` and :math:`\xi` through the per-ping-side Jacobians of
`bbf.inference.jacobian`, whose columns are the first variables in this order, so every row
above covers :math:`\xi` with no term of its own: the field has no data factor besides the
render it sits in.

With :math:`\psi` free (`Problem.psi_prior`), six more variables come last: per side the
beam centre, beam width and linear gain. Every row gains their derivatives: the beam's
through :math:`\partial S/\partial\psi` (`objective.beam_jacobian`, frozen with :math:`J`), the
gain's directly, :math:`\partial\log\bar I_b/\partial g_1 = \tfrac{\ln 10}{10} r_b`; their rows and
columns are dense, which six of them keep cheap.
"""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from bbf.inference.objective import Problem
from bbf.match.ratio import fisher_rows
from bbf.ping.factor import (
    DB_TO_LOG_I,
    background,
    bin_ranges,
    log_gain,
    log_gamma,
    predicted,
)
from bbf.ping.shadow import lit_probability
from bbf.ping.theta import with_psi


def layout(problem: Problem) -> dict:
    """Offsets of each variable group in the augmented vector; `nf` counts the field variables (d, xi) the Jacobians' columns cover."""
    n = problem.mesh.n_vertices
    n = problem.mesh.n_vertices
    nf = n + problem.n_xi
    nb = problem.prior.n_beta
    n_u = sum(2 * len(p) for p in problem.poses)
    u0 = np.cumsum([0] + [2 * len(p) for p in problem.poses])[:-1] + nf + nb
    n_psi = 6 if problem.free_psi else 0
    return {"n": n, "xi": n, "nf": nf, "beta": nf, "n_beta": nb, "u": nf + nb, "u_pass": u0, "psi": nf + nb + n_u,
            "n_psi": n_psi, "size": nf + nb + n_u + n_psi}


def _offset(problem: Problem, p: int) -> np.ndarray:
    """Log transmit power (n,) of the pings of pass p; zeros without `Problem.kappa_offset`."""
    if problem.kappa_offset is None:
        return np.zeros(len(problem.poses[p]))
    return np.asarray(problem.kappa_offset[p], dtype=float)


def ping_weights(problem: Problem, S, u_hat, uses):
    r"""Per pass (n, 2, B): Fisher weight of each scored bin and the two derivatives of :math:`\log\bar I`."""
    th, dr = problem.theta, problem.dr
    r = np.asarray(bin_ranges(problem.n_bins, dr))
    out = []
    for p, (y, Sp, up, use) in enumerate(zip(problem.y, S, u_hat, uses, strict=True)):
        W, a, c = np.zeros_like(Sp), np.zeros_like(Sp), np.zeros_like(Sp)
        off = _offset(problem, p)[:, None]
        for side in (0, 1):
            st = th.at(side)
            Ib, Isp = (np.asarray(x) for x in predicted(Sp[:, side], up[:, side, None], th, side, dr, off))
            ys = np.where(y[:, side] > 0, y[:, side], 1.0)
            lm = math.log1p(-st.eps) + np.asarray(log_gamma(ys, th.L, Ib))
            ls = math.log(st.eps) + np.asarray(log_gamma(ys, 1.0, Isp))
            resp = np.exp(lm - np.logaddexp(lm, ls))  # the main component's responsibility
            W[:, side] = np.where(use[:, side] & (y[:, side] > 0), problem.temper * th.L * resp, 0.0)
            ek = np.exp(up[:, side])[:, None]
            den = ek * Sp[:, side] + np.asarray(background(th, side, r, off))
            a[:, side] = ek / den
            c[:, side] = ek * Sp[:, side] / den
        out.append((W, a, c))
    return out


def shadow_rows(problem: Problem, S, u_hat):
    r"""Per pass (n, 2, B): coefficient of each bin in its dark block's :math:`\partial p_j/\partial S \,/\, (1 - p_j)`; zero in lit blocks.

    :math:`p_j` and :math:`\partial p_j / \partial \bar S_j` come from `bbf.ping.shadow.lit_probability`
    by one forward-mode derivative per ping.
    """
    th = problem.theta
    B = th.block
    out = []
    for p_i, (Sp, up) in enumerate(zip(S, u_hat, strict=True)):
        coef = np.zeros_like(Sp)
        nb = Sp.shape[-1] // B
        for side in (0, 1):
            def lit(s, u, o, side=side):
                return jax.jvp(lambda ss: lit_probability(ss, u, th, side, problem.dr, o), (s,), (jnp.ones_like(s),))
            p, dp = (np.asarray(x) for x in jax.vmap(lit)(jnp.asarray(Sp[:, side]), jnp.asarray(up[:, side]),
                                                          jnp.asarray(_offset(problem, p_i))))
            p = np.clip(p, 1e-12, 1 - 1e-12)  # as in `log_shadow_side`
            g = math.sqrt(problem.temper) * dp / ((1 - p) * B)  # per block, per bin of the block
            g = g * ~np.asarray(problem.labels[p_i][:, side, :nb], dtype=bool)  # dark blocks only
            if problem.valid is not None:
                g = g * problem.valid[p_i][:, side, :nb]
            coef[:, side, : nb * B] = np.repeat(g, B, axis=1)
        out.append(coef)
    return out


def assemble(problem: Problem, J: list[list[list[sp.csr_matrix]]], S, u_hat, Jb=None, psi=None, S0=None,
             fr=None, hz=None, prior=None, u_ratio=None) -> sp.csc_matrix:
    r"""The augmented GGN matrix :math:`H` (size from `layout`), symmetric positive semidefinite plus priors.

    prior: the depth prior whose precision goes in, default the problem's; e.g. the problem's with
    its heavy-tailed slope factor reweighted at the current depths (`bbf.prior.Prior.at`).

    J[p][t][c]: the frozen Jacobian of pass p, ping t, side c, (B, n) or (B, 2n) with the field.
    S, u_hat: the data term's rendered intensities and :math:`\hat u` per pass (n, 2).
    psi: the sonar parameters (2, 3) the weights are taken at (default theta's); Jb: per pass
    (n, 2, B, 2) beam derivatives of S (`objective.beam_jacobian`), needed when :math:`\psi` is free.

    S0: the render J and Jb were taken at. With it their rows are rescaled by :math:`S_b / S_{0,b}`,
    which freezes :math:`\partial \log S_b` instead of :math:`\partial S_b`. The reflectivity multiplies
    :math:`S`, so as :math:`\xi` moves by :math:`\delta` over a bin's footprint the bin's derivatives all
    grow by :math:`e^{\delta}`, and so does :math:`S_b`: the rescaled rows follow, the frozen ones do not.

    fr: the first-return factor's rows A (m, n) and weights W (m,) (`value_and_grad`'s details),
    adding :math:`A^\top W A` to the depth block; hz: the same for the horizon factor. With
    `Problem.horizon` set, the dark-block shadow rows are left out, as the data term leaves them out.

    u_ratio: the ratio match's gains per pass (n, 2) (`value_and_grad`'s details), needed when the
    problem has one; its curvature :math:`\tau D^\top (A^\top A - B^\top B) D`, with :math:`D` the bins'
    :math:`\partial \bar I` (`bbf.match.ratio.fisher_rows`), goes through the Jacobians of all passes
    together, and through the beam derivatives and the gain for :math:`\psi`. The rows are those of the
    held gains and shapes: with gains that follow the predictions (no `Problem.ratio_gain`) the
    path through them is missing here, though the gradient has it.
    """
    if psi is not None:
        problem = dataclasses.replace(problem, theta=with_psi(problem.theta, psi))
    free = problem.free_psi
    if free and Jb is None:
        raise ValueError("psi is free: pass the beam derivatives Jb")
    Jbs: list[np.ndarray] = [] if Jb is None else Jb
    lay = layout(problem)
    size = lay["size"]
    uses = problem.uses()
    pw = ping_weights(problem, S, u_hat, uses)
    sh = shadow_rows(problem, S, u_hat) if problem.horizon is None else None
    rows_i, rows_j, vals = [], [], []

    n, nf, ip = lay["n"], lay["nf"], lay["psi"]
    H_fpsi = np.zeros((nf, 6))  # dense blocks of the psi rows, added at the end
    H_xpsi = np.zeros((size - nf, 6))  # beta, u, psi rows against psi
    kg = DB_TO_LOG_I * np.asarray(bin_ranges(problem.n_bins, problem.dr))  # :math:`\partial \log g / \partial g_1` per bin

    def add(Msp):
        Mc = Msp.tocoo()
        rows_i.append(Mc.row)
        rows_j.append(Mc.col)
        vals.append(Mc.data)

    B, nbin = problem.theta.block, problem.n_bins
    nb = nbin // B
    in_block = np.arange(nbin) < nb * B
    J_all, dS_all, gain_all = [], [], []  # per pass, for the ratio match's rows across passes
    std = np.array([problem.theta.at(side).log_kappa_std for side in (0, 1)])
    for p, (W, a, c) in enumerate(pw):
        # every bin of the pass as one row: ping-major, port then starboard, as the arrays (n, 2, B) ravel
        n_ps = 2 * len(problem.poses[p])
        Jp = sp.csr_matrix(sp.vstack([J[p][t][side] for t in range(n_ps // 2) for side in (0, 1)]))
        if Jp.get_shape()[1] != nf:
            raise ValueError(f"Jacobian has {Jp.get_shape()[1]} columns, the problem {nf} field variables")
        w, a_, c_ = W.ravel(), a.ravel(), c.ravel()
        row = np.arange(len(w))
        scale = np.ones(len(w))
        if S0 is not None:
            s_now, s_then = S[p].ravel(), S0[p].ravel()
            ok = (s_now > 0) & (s_then > 0)
            scale = np.where(ok, np.clip(s_now / np.where(ok, s_then, 1.0), 1e-3, 1e3), 1.0)
            Jp = sp.csr_matrix(sp.diags(scale) @ Jp)
        J_all.append(Jp)
        ps = row // nbin  # ping-side of each row
        u0 = lay["u_pass"][p]
        # ping bins: field-field, field-u, u-u
        add(Jp.T @ sp.diags(w * a_**2) @ Jp)
        cross = sp.coo_matrix(Jp.T @ sp.csr_matrix((w * a_ * c_, (row, ps)), shape=(len(w), n_ps)))
        add(sp.coo_matrix((cross.data, (cross.row, cross.col + u0)), shape=(size, size)))
        add(sp.coo_matrix((cross.data, (cross.col + u0, cross.row)), shape=(size, size)))
        h_uu = np.bincount(ps, w * c_**2, n_ps) + np.tile(1.0 / std**2, n_ps // 2)
        add(sp.coo_matrix((h_uu, (u0 + np.arange(n_ps), u0 + np.arange(n_ps))), shape=(size, size)))
        # shadow blocks: one row per block, sum of its bins' J rows times the coefficient
        agg = G = None
        if sh is not None:
            keep = np.tile(in_block, n_ps)
            block = ps * nb + np.minimum(row % nbin // B, nb - 1)
            agg = sp.csr_matrix((sh[p].ravel()[keep], (block[keep], row[keep])), shape=(n_ps * nb, len(w)))
            G = agg @ Jp
            add(G.T @ G)
        if free:
            # psi columns of each bin, in its own side's three: :math:`a\,\partial S/\partial(\beta_0, \omega)` and :math:`\partial \log g/\partial g_1`
            own = np.repeat(np.eye(2, dtype=bool)[ps % 2], 3, axis=1)  # (rows, 6): the bin's side's columns
            dS = np.where(own, np.tile(np.column_stack([scale[:, None] * Jbs[p].reshape(-1, 2), np.zeros(len(w))]), 2), 0.0)
            gain = np.where(own, np.tile(np.tile(kg, n_ps)[:, None], 6), 0.0) * [0, 0, 1, 0, 0, 1]
            Gp = a_[:, None] * dS + gain
            dS_all.append(dS)
            gain_all.append(gain)
            H_fpsi += Jp.T @ ((w * a_)[:, None] * Gp)
            H_xpsi[u0 - nf: u0 - nf + n_ps] += np.column_stack([np.bincount(ps, w * c_ * Gp[:, k], n_ps) for k in range(6)])
            H_xpsi[ip - nf: ip - nf + 6] += Gp.T @ (w[:, None] * Gp)
            if agg is not None and G is not None:
                Gs = agg @ dS  # the gain does not enter p_j
                H_fpsi += G.T @ Gs
                H_xpsi[ip - nf: ip - nf + 6] += Gs.T @ Gs
    if problem.ratio is not None and problem.ratio.n_units > 0:
        # ratio match: :math:`\tau D^\top (A^\top A - B^\top B) D` with D the bins' :math:`\partial \bar I`, through the
        # Jacobians for (d, xi) and, with psi free, through the beam derivatives and the gain
        if u_ratio is None:
            raise ValueError("the problem has a ratio match: pass its gains u_ratio")
        th, r = problem.theta, np.asarray(bin_ranges(nbin, problem.dr))
        Ibar, dI = [], []
        for p, (Sp, ur) in enumerate(zip(S, u_ratio, strict=True)):
            off = _offset(problem, p)[:, None]
            ib, di = np.zeros_like(Sp), np.zeros_like(Sp)
            for side in (0, 1):
                ib[:, side] = np.asarray(predicted(Sp[:, side], ur[:, side, None], th, side, problem.dr, off)[0])
                di[:, side] = np.exp(np.asarray(log_gain(r, th.at(side).gain_db)) + ur[:, side, None])  # :math:`g_c(r_b) e^u`
            Ibar.append(ib)
            dI.append(di)
        A, Bm = fisher_rows(Ibar, problem.ratio)
        A, Bm = math.sqrt(problem.temper) * A, math.sqrt(problem.temper) * Bm
        dIv = np.concatenate([x.ravel() for x in dI])
        Jst = sp.csr_matrix(sp.vstack(J_all))
        GA, GB = (A @ sp.diags(dIv)) @ Jst, (Bm @ sp.diags(dIv)) @ Jst
        add(GA.T @ GA)
        add(-(GB.T @ GB))
        if free:
            # :math:`\partial \bar I_b / \partial \psi`: the beam's through S, and the gain's, which scales all of
            # :math:`\bar I_b` (the background is behind the gain too)
            Iv = np.concatenate([x.ravel() for x in Ibar])
            D = dIv[:, None] * np.vstack(dS_all) + Iv[:, None] * np.vstack(gain_all)  # bins x psi
            PA, PB = A @ D, Bm @ D
            H_fpsi += GA.T @ PA - GB.T @ PB
            H_xpsi[ip - nf: ip - nf + 6] += PA.T @ PA - PB.T @ PB
    if free:
        H_xpsi[ip - nf: ip - nf + 6] = 0.5 * (H_xpsi[ip - nf: ip - nf + 6] + H_xpsi[ip - nf: ip - nf + 6].T)
        dpsi = sp.coo_matrix(np.vstack([H_fpsi, H_xpsi]))  # rows: every variable, columns: psi
        dpsi = sp.coo_matrix((dpsi.data, (dpsi.row, dpsi.col + ip)), shape=(size, size))
        add(dpsi)
        add(dpsi.T)  # the psi-psi block now counts twice: halve it below
        blk = H_xpsi[ip - nf: ip - nf + 6]
        ii, jj = np.meshgrid(np.arange(ip, ip + 6), np.arange(ip, ip + 6), indexing="ij")
        add(sp.coo_matrix((-blk.ravel(), (ii.ravel(), jj.ravel())), shape=(size, size)))
        assert problem.psi_prior is not None
        sd = np.asarray(problem.psi_prior[1], float).ravel()
        add(sp.coo_matrix((1.0 / sd**2, (np.arange(ip, ip + 6), np.arange(ip, ip + 6))), shape=(size, size)))
    # prior on (d, beta), with beta moved behind the field variables, and on xi
    Pr = (problem.prior if prior is None else prior).joint_precision().tocoo()
    at = np.concatenate([np.arange(n), lay["beta"] + np.arange(lay["n_beta"])])
    add(sp.coo_matrix((Pr.data, (at[Pr.row], at[Pr.col])), shape=(size, size)))
    for rows in (fr, hz):
        if rows is not None:
            A, W = rows
            Hf = (A.T @ sp.diags(W) @ A).tocoo()
            add(sp.coo_matrix((Hf.data, (Hf.row, Hf.col)), shape=(size, size)))
    if problem.refl is not None:
        Lx = problem.refl.Lambda.tocoo()
        add(sp.coo_matrix((Lx.data, (Lx.row + lay["xi"], Lx.col + lay["xi"])), shape=(size, size)))
    return sp.csc_matrix(sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(rows_j))),
                                       shape=(size, size)))
