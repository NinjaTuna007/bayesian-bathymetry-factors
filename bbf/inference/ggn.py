r"""Generalised Gauss--Newton curvature of the log posterior as a sparse augmented matrix.

Variables, in order: depths :math:`d` (n), trend :math:`\beta` (3, or 1 for a constant mean), one :math:`u = \log\kappa` per
ping-side (all passes, ping-major, port then starboard), and one side offset :math:`\mu` per
pass pair. Keeping :math:`u` and :math:`\mu` as variables (instead of eliminating them) gives
the same Gauss--Newton step and Laplace marginal for :math:`d`, while :math:`H` keeps :math:`d`--:math:`d`
entries only between vertices that share a bin.

Each data term contributes :math:`G^\top I G`, with :math:`I` the Fisher information of its
likelihood in a scalar prediction and :math:`G` that prediction's derivative:

* ping bin :math:`b` (bins the ping factor scores): prediction :math:`\log\bar I_b`, Fisher
  :math:`L\,r_b` with :math:`r_b` the main component's responsibility,
  :math:`\partial\log\bar I_b/\partial S_b = e^u / (e^u S_b + N)`,
  :math:`\partial\log\bar I_b/\partial u = e^u S_b / (e^u S_b + N)`;
* dark shadow block :math:`j` (lit blocks are left to the ping factor): prediction :math:`p_j`,
  weight :math:`1/(1 - p_j)^2`, the curvature of :math:`-\log(1 - p_j)` in :math:`p_j`, through the
  block mean of :math:`S` with :math:`\hat\kappa` held fixed;
* match pair :math:`m`: prediction :math:`e_m - \sigma_m\mu`, Fisher :math:`r_m L^2/(2L+1)` of
  :math:`p_L`, :math:`\partial e/\partial S_a = 1/(S_a + f_a)`, :math:`\partial e/\partial S_b = -1/(S_b + f_b)`;

plus the prior's exact precision on :math:`(d, \beta)` and the Gaussian prior on each :math:`u`.

With :math:`\psi` free (`Problem.psi_prior`), six more variables follow :math:`\mu`: per side the
beam centre, beam width and linear gain. Every row gains their derivatives: the beam's
through :math:`\partial S/\partial\psi` (`objective.beam_jacobian`, frozen with :math:`J`), the
gain's directly, :math:`\partial\log\bar I_b/\partial g_1 = \tfrac{\ln 10}{10} r_b` and
:math:`\partial e_m/\partial g_{1} = \tfrac{\ln 10}{10}(r_a - r_b)` on the two sides of a pair; their
rows and columns are dense, which six of them keep cheap.
:math:`S` reaches :math:`d` through the per-ping-side Jacobians of `bbf.inference.jacobian`.
"""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from bbf.inference.objective import Problem
from bbf.match.factor import log_pair, log_ratio_pdf, noise_floor, pair_eps, residual
from bbf.ping.factor import DB_TO_LOG_I, bin_ranges, log_gamma, predicted
from bbf.ping.shadow import lit_probability
from bbf.ping.theta import with_psi


def layout(problem: Problem) -> dict:
    """Offsets of each variable group in the augmented vector."""
    n = problem.mesh.n_vertices
    nb = problem.prior.n_beta
    n_u = sum(2 * len(p) for p in problem.poses)
    keys = sorted(problem.pairs)
    u0 = np.cumsum([0] + [2 * len(p) for p in problem.poses])[:-1] + n + nb
    mu = n + nb + n_u
    n_psi = 6 if problem.free_psi else 0
    return {"n": n, "beta": n, "n_beta": nb, "u": n + nb, "u_pass": u0, "mu": mu, "keys": keys,
            "psi": mu + len(keys), "n_psi": n_psi, "size": mu + len(keys) + n_psi}


def ping_weights(problem: Problem, S, u_hat, uses):
    r"""Per pass (n, 2, B): Fisher weight of each scored bin and the two derivatives of :math:`\log\bar I`."""
    th, dr = problem.theta, problem.dr
    out = []
    for y, Sp, up, use in zip(problem.y, S, u_hat, uses, strict=True):
        W, a, c = np.zeros_like(Sp), np.zeros_like(Sp), np.zeros_like(Sp)
        for side in (0, 1):
            st = th.at(side)
            Ib, Isp = (np.asarray(x) for x in predicted(Sp[:, side], up[:, side, None], th, side, dr))
            ys = np.where(y[:, side] > 0, y[:, side], 1.0)
            lm = math.log1p(-st.eps) + np.asarray(log_gamma(ys, th.L, Ib))
            ls = math.log(st.eps) + np.asarray(log_gamma(ys, 1.0, Isp))
            resp = np.exp(lm - np.logaddexp(lm, ls))  # the main component's responsibility
            W[:, side] = np.where(use[:, side] & (y[:, side] > 0), th.L * resp, 0.0)
            ek = np.exp(up[:, side])[:, None]
            den = ek * Sp[:, side] + st.noise
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
            def lit(s, u, side=side):
                return jax.jvp(lambda ss: lit_probability(ss, u, th, side), (s,), (jnp.ones_like(s),))
            p, dp = (np.asarray(x) for x in jax.vmap(lit)(jnp.asarray(Sp[:, side]), jnp.asarray(up[:, side])))
            p = np.clip(p, 1e-12, 1 - 1e-12)  # as in `log_shadow_side`
            g = dp / ((1 - p) * B)  # per block, per bin of the block
            g = g * ~np.asarray(problem.labels[p_i][:, side, :nb], dtype=bool)  # dark blocks only
            if problem.valid is not None:
                g = g * problem.valid[p_i][:, side, :nb]
            coef[:, side, : nb * B] = np.repeat(g, B, axis=1)
        out.append(coef)
    return out


def match_rows(problem: Problem, S, mus):
    r"""Per pair key: (:math:`\sqrt{I_m}`, :math:`\partial e/\partial S_a`, :math:`\partial e/\partial S_b`) per pair, at :math:`\hat\mu`."""
    th = problem.theta
    floor = noise_floor(th)
    L = th.L
    out = {}
    for key, mu in zip(sorted(problem.pairs), mus, strict=True):
        a, b = key
        pr = problem.pairs[key]
        e = np.asarray(residual(pr, S[a], S[b], problem.y[a], problem.y[b], th, problem.dr)) - pr.sign * mu
        eps = pair_eps(pr, th)
        lp = np.log1p(-eps) + np.asarray(log_ratio_pdf(e, L))
        resp = np.exp(lp - np.asarray(log_pair(e, eps, L)))  # the main component's responsibility
        Sa = S[a][pr.ping_a, pr.side_a, pr.bin_a]
        Sb = S[b][pr.ping_b, pr.side_b, pr.bin_b]
        out[key] = (np.sqrt(resp * L**2 / (2 * L + 1)), 1.0 / (Sa + floor[pr.side_a]), -1.0 / (Sb + floor[pr.side_b]))
    return out


def assemble(problem: Problem, J: list[list[list[sp.csr_matrix]]], S, u_hat, mus, Jb=None, psi=None) -> sp.csc_matrix:
    r"""The augmented GGN matrix :math:`H` (size from `layout`), symmetric positive semidefinite plus priors.

    J[p][t][c]: the frozen Jacobian of pass p, ping t, side c (B, n). S, u_hat, mus: the data
    term's rendered intensities, :math:`\hat u` per pass (n, 2) and :math:`\hat\mu` per pair key.
    psi: the sonar parameters (2, 3) the weights are taken at (default theta's); Jb: per pass
    (n, 2, B, 2) beam derivatives of S (`objective.beam_jacobian`), needed when :math:`\psi` is free.
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
    sh = shadow_rows(problem, S, u_hat)
    rows_i, rows_j, vals = [], [], []

    n, ip = lay["n"], lay["psi"]
    H_dpsi = np.zeros((n, 6))  # dense blocks of the psi rows, added at the end
    H_xpsi = np.zeros((size - n, 6))  # beta, u, mu, psi rows against psi
    kg = DB_TO_LOG_I * np.asarray(bin_ranges(problem.n_bins, problem.dr))  # :math:`\partial \log g / \partial g_1` per bin

    def add(Msp):
        Mc = Msp.tocoo()
        rows_i.append(Mc.row)
        rows_j.append(Mc.col)
        vals.append(Mc.data)

    B = problem.theta.block
    for p, (W, a, c) in enumerate(pw):
        for t in range(len(problem.poses[p])):
            for side in (0, 1):
                Jt = J[p][t][side]
                w = W[t, side]
                ui = lay["u_pass"][p] + 2 * t + side
                # ping bins: d-d, d-u, u-u
                da = w * a[t, side] ** 2
                add(Jt.T @ sp.diags(da) @ Jt)
                h_du = Jt.T @ (w * a[t, side] * c[t, side])
                nz = np.flatnonzero(h_du)
                add(sp.coo_matrix((h_du[nz], (nz, np.full(len(nz), ui))), shape=(size, size)))
                add(sp.coo_matrix((h_du[nz], (np.full(len(nz), ui), nz)), shape=(size, size)))
                h_uu = float(np.sum(w * c[t, side] ** 2)) + 1.0 / problem.theta.at(side).log_kappa_std ** 2
                add(sp.coo_matrix(([h_uu], ([ui], [ui])), shape=(size, size)))
                if free:
                    # psi columns of this side's bins: :math:`a\,\partial S/\partial(\beta_0, \omega)` and :math:`\partial \log g/\partial g_1`
                    Gp = np.column_stack([a[t, side][:, None] * Jbs[p][t, side], kg])
                    cols = slice(3 * side, 3 * side + 3)
                    H_dpsi[:, cols] += Jt.T @ ((w * a[t, side])[:, None] * Gp)
                    H_xpsi[ui - n, cols] += (w * c[t, side]) @ Gp
                    H_xpsi[ip - n + 3 * side: ip - n + 3 * side + 3, cols] += Gp.T @ (w[:, None] * Gp)
                # shadow blocks: one row per block, sum of its bins' J rows times the coefficient
                coef = sh[p][t, side]
                nb = len(coef) // B
                agg = sp.coo_matrix((coef[: nb * B], (np.repeat(np.arange(nb), B), np.arange(nb * B))),
                                    shape=(nb, B * nb + (len(coef) - nb * B))).tocsr()
                G = agg @ Jt
                add(G.T @ G)
                if free:
                    Gs = agg @ np.column_stack([Jbs[p][t, side], np.zeros(len(coef))])  # the gain does not enter p_j
                    cols = slice(3 * side, 3 * side + 3)
                    H_dpsi[:, cols] += G.T @ Gs
                    H_xpsi[ip - n + 3 * side: ip - n + 3 * side + 3, cols] += Gs.T @ Gs
    # match pairs: rows over d and mu
    for k, (key, (sq, ca, cb)) in enumerate(match_rows(problem, S, mus).items()):
        a_, b_ = key
        pr = problem.pairs[key]
        rows = [sq[m] * (ca[m] * J[a_][pr.ping_a[m]][pr.side_a[m]][pr.bin_a[m]]
                         + cb[m] * J[b_][pr.ping_b[m]][pr.side_b[m]][pr.bin_b[m]]) for m in range(len(pr))]
        if not rows:
            continue
        Gd = sp.vstack(rows).tocsr()
        gmu = -sq * pr.sign
        mi = lay["mu"] + k
        add(Gd.T @ Gd)
        cross = Gd.T @ gmu
        nz = np.flatnonzero(cross)
        add(sp.coo_matrix((cross[nz], (nz, np.full(len(nz), mi))), shape=(size, size)))
        add(sp.coo_matrix((cross[nz], (np.full(len(nz), mi), nz)), shape=(size, size)))
        add(sp.coo_matrix(([float(gmu @ gmu)], ([mi], [mi])), shape=(size, size)))
        if free:
            Gp = np.zeros((len(pr), 6))
            for m in range(len(pr)):
                sa, sb = 3 * pr.side_a[m], 3 * pr.side_b[m]
                Gp[m, sa:sa + 2] += ca[m] * Jbs[a_][pr.ping_a[m], pr.side_a[m], pr.bin_a[m]]
                Gp[m, sb:sb + 2] += cb[m] * Jbs[b_][pr.ping_b[m], pr.side_b[m], pr.bin_b[m]]
                Gp[m, sa + 2] += kg[pr.bin_a[m]]  # e has :math:`+\log g_a`
                Gp[m, sb + 2] -= kg[pr.bin_b[m]]  # and :math:`-\log g_b`
            Gp *= sq[:, None]
            H_dpsi += Gd.T @ Gp
            H_xpsi[mi - n] += gmu @ Gp
            H_xpsi[ip - n: ip - n + 6] += Gp.T @ Gp
    if free:
        H_xpsi[ip - n: ip - n + 6] = 0.5 * (H_xpsi[ip - n: ip - n + 6] + H_xpsi[ip - n: ip - n + 6].T)
        dpsi = sp.coo_matrix(np.vstack([H_dpsi, H_xpsi]))  # rows: every variable, columns: psi
        dpsi = sp.coo_matrix((dpsi.data, (dpsi.row, dpsi.col + ip)), shape=(size, size))
        add(dpsi)
        add(dpsi.T)  # the psi-psi block now counts twice: halve it below
        blk = H_xpsi[ip - n: ip - n + 6]
        ii, jj = np.meshgrid(np.arange(ip, ip + 6), np.arange(ip, ip + 6), indexing="ij")
        add(sp.coo_matrix((-blk.ravel(), (ii.ravel(), jj.ravel())), shape=(size, size)))
        assert problem.psi_prior is not None
        sd = np.asarray(problem.psi_prior[1], float).ravel()
        add(sp.coo_matrix((1.0 / sd**2, (np.arange(ip, ip + 6), np.arange(ip, ip + 6))), shape=(size, size)))
    # prior on (d, beta)
    add(problem.prior.joint_precision())
    # an offset no pair constrains (no mixed pairs) needs a tiny ridge to keep H definite
    ridge = np.arange(lay["mu"], lay["psi"])
    add(sp.coo_matrix((np.full(len(ridge), 1e-4), (ridge, ridge)), shape=(size, size)))
    return sp.csc_matrix(sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(rows_j))),
                                       shape=(size, size)))
