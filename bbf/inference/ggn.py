r"""Generalised Gauss--Newton curvature of the log posterior as a sparse augmented matrix.

Variables, in order: depths :math:`d` (n), trend :math:`\beta` (3), one :math:`u = \log\kappa` per
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
* shadow block :math:`j`: prediction :math:`p_j`, Fisher :math:`1/(p_j(1 - p_j))`, through the block
  mean of :math:`S` with :math:`\hat\kappa` held fixed;
* match pair :math:`m`: prediction :math:`e_m - \sigma_m\mu`, Fisher :math:`r_m L^2/(2L+1)` of
  :math:`p_L`, :math:`\partial e/\partial S_a = 1/(S_a + f_a)`, :math:`\partial e/\partial S_b = -1/(S_b + f_b)`;

plus the prior's exact precision on :math:`(d, \beta)` and the Gaussian prior on each :math:`u`.
:math:`S` reaches :math:`d` through the per-ping-side Jacobians of `bbf.inference.jacobian`.
"""
import math

import numpy as np
import scipy.sparse as sp
from scipy.special import gammainc, gammaln

from bbf.inference.objective import Problem
from bbf.match.factor import E_MAX, log_ratio_pdf
from bbf.ping.factor import DB_TO_LOG_I, bin_ranges
from bbf.ping.shadow import noise_lit_rate, threshold


def layout(problem: Problem) -> dict:
    """Offsets of each variable group in the augmented vector."""
    n = problem.mesh.n_vertices
    n_u = sum(2 * len(p) for p in problem.poses)
    keys = sorted(problem.pairs)
    u0 = np.cumsum([0] + [2 * len(p) for p in problem.poses])[:-1] + n + 3
    return {"n": n, "beta": n, "u": n + 3, "u_pass": u0, "mu": n + 3 + n_u, "keys": keys,
            "size": n + 3 + n_u + len(keys)}


def _gain(theta, side, n_bins, dr):
    r = np.asarray(bin_ranges(n_bins, dr))
    g = np.asarray(theta.gain_db)[side]
    return np.exp(DB_TO_LOG_I * (g[0] + g[1] * r + g[2] * np.log10(r)))


def _log_gamma(y, k, mean):
    return k * math.log(k) - gammaln(k) + (k - 1) * np.log(y) - k * np.log(mean) - k * y / mean


def ping_weights(problem: Problem, S, u_hat, uses):
    r"""Per pass (n, 2, B): Fisher weight of each scored bin and the two derivatives of :math:`\log\bar I`."""
    th, dr = problem.theta, problem.dr
    out = []
    for y, Sp, up, use in zip(problem.y, S, u_hat, uses):
        W, a, c = np.zeros_like(Sp), np.zeros_like(Sp), np.zeros_like(Sp)
        for side in (0, 1):
            st = th.at(side)
            g = _gain(th, side, Sp.shape[-1], dr)
            ek = np.exp(up[:, side])[:, None]
            Ib = g * (ek * Sp[:, side] + st.noise)
            r = np.asarray(bin_ranges(Sp.shape[-1], dr))
            Isp = Ib + g * (st.eta + st.zeta * np.exp(-r / st.ell))
            ys = np.where(y[:, side] > 0, y[:, side], 1.0)
            lm = math.log1p(-st.eps) + _log_gamma(ys, th.L, Ib)
            ls = math.log(st.eps) + _log_gamma(ys, 1.0, Isp)
            resp = np.exp(lm - np.logaddexp(lm, ls))
            ok = use[:, side] & (y[:, side] > 0)
            W[:, side] = np.where(ok, th.L * resp, 0.0)
            den = ek * Sp[:, side] + st.noise
            a[:, side] = ek / den
            c[:, side] = ek * Sp[:, side] / den
        out.append((W, a, c))
    return out


def shadow_rows(problem: Problem, S, u_hat):
    r"""Per pass (n, 2, B): coefficient of each bin in its block's :math:`\sqrt{I_j}\,\partial p_j/\partial S`."""
    th = problem.theta
    B = th.block
    k = B * th.L
    T = threshold(th)
    fa = th.label_fa
    out = []
    for p_i, (Sp, up) in enumerate(zip(S, u_hat)):
        coef = np.zeros_like(Sp)
        nb = Sp.shape[-1] // B
        for side in (0, 1):
            st = th.at(side)
            kap = np.exp(up[:, side])[:, None]
            Sbar = Sp[:, side, : nb * B].reshape(len(Sp), nb, B).mean(axis=2)
            m = 1.0 + kap * Sbar / st.noise
            x = k * T / m
            D = 1.0 - gammainc(k, x)
            p0 = noise_lit_rate(th, side, nb)[None, :]
            p = np.clip(p0 + (1 - p0) * (D - fa) / (1 - fa), 1e-12, 1 - 1e-12)
            dD_dm = np.exp((k - 1) * np.log(x) - x - gammaln(k)) * k * T / m**2
            dp_dm = (1 - p0) / (1 - fa) * dD_dm
            g = np.sqrt(1.0 / (p * (1 - p))) * dp_dm * kap / (st.noise * B)  # per block, per bin of the block
            if problem.valid is not None:
                g = g * problem.valid[p_i][:, side, :nb]
            coef[:, side, : nb * B] = np.repeat(g, B, axis=1)
        out.append(coef)
    return out


def match_rows(problem: Problem, S, mus):
    r"""Per pair key: (:math:`\sqrt{I_m}`, :math:`\partial e/\partial S_a`, :math:`\partial e/\partial S_b`) per pair, at :math:`\hat\mu`."""
    th = problem.theta
    floor = np.asarray(th.noise) * np.exp(-np.asarray(th.log_kappa_mean))
    L = th.L
    out = {}
    for key, mu in zip(sorted(problem.pairs), mus):
        a, b = key
        pr = problem.pairs[key]
        Sa = S[a][pr.ping_a, pr.side_a, pr.bin_a]
        Sb = S[b][pr.ping_b, pr.side_b, pr.bin_b]
        ga = _gain_at(th, pr.side_a, pr.bin_a, problem.dr)
        gb = _gain_at(th, pr.side_b, pr.bin_b, problem.dr)
        ya = problem.y[a][pr.ping_a, pr.side_a, pr.bin_a]
        yb = problem.y[b][pr.ping_b, pr.side_b, pr.bin_b]
        e = (np.log(yb) - np.log(ya) - (np.log(gb * (Sb + floor[pr.side_b])) - np.log(ga * (Sa + floor[pr.side_a])))
             - pr.sign * mu)
        eps = np.clip(np.asarray(th.eps)[pr.side_a] + np.asarray(th.eps)[pr.side_b], 1e-12, 0.5)
        lp = np.log1p(-eps) + np.asarray(log_ratio_pdf(e, L))
        lu = np.where(np.abs(e) <= E_MAX, -math.log(2 * E_MAX), -1e3) + np.log(eps)
        resp = np.exp(lp - np.logaddexp(lp, lu))
        out[key] = (np.sqrt(resp * L**2 / (2 * L + 1)), 1.0 / (Sa + floor[pr.side_a]), -1.0 / (Sb + floor[pr.side_b]))
    return out


def _gain_at(theta, side, b, dr):
    r = np.asarray(bin_ranges(int(b.max()) + 2 if len(b) else 2, dr))[b]
    g = np.asarray(theta.gain_db)[side]
    return np.exp(DB_TO_LOG_I * (g[:, 0] + g[:, 1] * r + g[:, 2] * np.log10(r)))


def assemble(problem: Problem, J: list[list[list[sp.csr_matrix]]], S, u_hat, mus) -> sp.csc_matrix:
    r"""The augmented GGN matrix :math:`H` (size from `layout`), symmetric positive semidefinite plus priors.

    J[p][t][c]: the frozen Jacobian of pass p, ping t, side c (B, n). S, u_hat, mus: the data
    term's rendered intensities, :math:`\hat u` per pass (n, 2) and :math:`\hat\mu` per pair key.
    """
    lay = layout(problem)
    size = lay["size"]
    uses = problem.uses()
    pw = ping_weights(problem, S, u_hat, uses)
    sh = shadow_rows(problem, S, u_hat)
    rows_i, rows_j, vals = [], [], []

    def add(Msp, r0=0, c0=0):
        Mc = Msp.tocoo()
        rows_i.append(Mc.row + r0)
        rows_j.append(Mc.col + c0)
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
                # shadow blocks: one row per block, sum of its bins' J rows times the coefficient
                coef = sh[p][t, side]
                nb = len(coef) // B
                agg = sp.coo_matrix((coef[: nb * B], (np.repeat(np.arange(nb), B), np.arange(nb * B))),
                                    shape=(nb, B * nb + (len(coef) - nb * B))).tocsr()
                G = agg @ Jt
                add(G.T @ G)
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
    # prior on (d, beta)
    add(problem.prior.joint_precision())
    # an offset no pair constrains (no mixed pairs) needs a tiny ridge to keep H definite
    ridge = np.arange(lay["mu"], size)
    add(sp.coo_matrix((np.full(len(ridge), 1e-8), (ridge, ridge)), shape=(size, size)))
    return sp.csc_matrix(sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(rows_j))),
                                       shape=(size, size)))
