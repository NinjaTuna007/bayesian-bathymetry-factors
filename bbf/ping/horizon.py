r"""Horizon factor: lit/shadow labels against the line-of-sight margin of the predicted seabed.

A shadow's length is set by the geometry alone: the sonar's pose, the crest that casts it and
the seabed behind. On a flat floor a crest of height :math:`h` at horizontal distance :math:`x_c`
from a sonar at altitude :math:`A` casts a shadow :math:`\ell = h\,x_c / (A - h)` long, so the
shadows of one object seen from several ranges tie its height to the depth of the floor, and a
tilt of the floor behind it moves where its shadow ends. The intensity-based dark-block term of
`bbf.ping.shadow` sees this only through the diffracted visibility, which changes within a
Fresnel zone (centimetres) of the edge: a predicted shadow edge half a metre off gets no pull,
and one that runs too long costs only the spike-bounded loss of the lit bins it darkens.

This factor scores the labels on a geometric margin instead. For curve sample :math:`k`
(`geometry.trace`) with elevation :math:`\alpha_k` seen from the sonar, the crest :math:`j^*` of
`geometry.horizon_visibility` rises

.. math::

    h_k = d_1 \sin(\alpha_{j^*} - \alpha_k)

above the line of sight to it, :math:`d_1` the crest's range: :math:`h_k > 0` in shadow,
:math:`h_k < 0` in the open. That crest is at least the self-shadow margin :math:`\delta`
(`Theta.shadow_margin`) closer, so it misses a face turned away from the sonar, which the render
darkens through its incidence (:math:`\cos\varphi_k \le 0`); the face's own tangent plane rises
:math:`-\delta \cos\varphi_k` above the line of sight over :math:`\delta`, and the margin is the soft
maximum (temperature :math:`\tau`) of the two. On open ground both are negative, about
:math:`-\delta \sin\gamma` at grazing angle :math:`\gamma`. A range block :math:`j` sees the seabed if any of its samples does,
so its margin is the soft minimum over the samples whose range falls in it,

.. math::

    m_j = -\tau \log \frac{\sum_k \pi_{kj}\, e^{-h_k / \tau}}{\sum_k \pi_{kj}},

with :math:`\pi_{kj}` the linear split of sample :math:`k` between the two nearest block centres
(as `render.split_into_bins` splits it between bins), so :math:`m_j` moves smoothly with both the
heights and the ranges. The label is then

.. math::

    p(\ell_j = \text{lit} \mid d) = p_{0,j} + (1 - p_{0,j})(1 - \epsilon)\,
        \Phi\!\left(-\frac{m_j}{\sigma}\right),

with :math:`p_{0,j}` the lit rate without a seabed echo (`shadow.noise_lit_rate`), :math:`\epsilon`
the chance that visible seabed is labelled dark (a weak or dropped-out echo), and :math:`\sigma`
the height above the line of sight at which a block is as likely shadowed as not. Every scored
block has a gradient, however far its predicted edge is from the data's, and a shadow too long
costs as much as one too short.

Which blocks are scored (`scored_blocks`) is fixed per round from the data: blocks past the
first echo, lit ones, and dark ones followed by lit seabed within `max_gap`, so a dark run is a
shadow with a far end and not the beam fading at far range. Blocks with no predicted seabed
sample (nearer than the predicted nearest range) are left to the first-return factor.

The lit labels are thresholded from the intensities the ping factor scores, so the two count
the same bins twice; the label carries one bit per block against the ping factor's density of
every bin, and its weight is mainly where the two disagree on whether the seabed is seen.

Derivatives hold the curve's horizontal samples fixed (exact for a level ping plane, as in
`bbf.inference.jacobian`): :math:`h_k` then depends on the three depths of the sample's triangle
(its height and face) and the three of its crest's, and the range :math:`r_k` on the first three. The rows
:math:`\partial m_j / \partial d` are assembled from those, giving both the gradient and the
Gauss--Newton rows :math:`A^\top W A`, :math:`W` the Fisher weight of the Bernoulli label in :math:`m_j`.
"""
from dataclasses import dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from bbf import mesh as M
from bbf.ping import geometry as G
from bbf.ping.shadow import noise_lit_rate
from bbf.ping.theta import Theta

SIGMA, EPS, TAU, MAX_GAP, DEN_MIN, ROW_TOL = 0.1, 0.05, 0.02, 15.0, 0.5, 1e-3
H_OPEN = 1.0  # margin (m) of samples with no crest before them: in the open


@dataclass(frozen=True, eq=False)
class Horizon:
    r"""The factor's constants and, per round, the blocks it scores.

    use: per pass (n, 2, n_blocks), from `scored_blocks` (None: no block yet); sigma: margin
    scale (m); eps: chance visible seabed is labelled dark; tau: soft-min temperature (m);
    max_gap: longest dark run (m) scored as shadow, which must end in lit seabed;
    den_min: least sample weight a block needs to have a predicted seabed; row_tol: row entries
    below this fraction of the row's largest are left out of the Gauss--Newton rows (not the gradient).
    """
    use: list[np.ndarray] | None = None
    sigma: float = SIGMA
    eps: float = EPS
    tau: float = TAU
    max_gap: float = MAX_GAP
    den_min: float = DEN_MIN
    row_tol: float = ROW_TOL


def scored_blocks(labels, valid, r0, block_dr: float, max_gap: float = MAX_GAP) -> list[np.ndarray]:
    """Per pass (n, 2, n_blocks): valid blocks past the first echo that are lit, or dark with lit seabed within max_gap beyond.

    labels: per pass (n, 2, n_blocks); valid: per pass, same shape, or None; r0: per pass (n, 2)
    first echoes (NaN: from the first lit block on).
    """
    out = []
    for p, lab in enumerate(labels):
        lab = np.asarray(lab, dtype=bool)
        nb = lab.shape[-1]
        rb = (np.arange(nb) + 0.5) * block_dr
        first = np.where(lab.any(-1), lab.argmax(-1), nb)
        start = (first + 0.5) * block_dr
        if r0 is not None:
            r = np.asarray(r0[p], dtype=float)
            start = np.where(np.isfinite(r), r, start)
        past = rb >= start[..., None]
        # distance to the next lit block beyond each block, by a reverse running minimum of lit indices
        idx = np.where(lab, np.arange(nb), nb)
        nxt = np.minimum.accumulate(idx[..., ::-1], axis=-1)[..., ::-1]
        nxt = np.concatenate([nxt[..., 1:], np.full(nxt.shape[:-1] + (1,), nb)], axis=-1)
        closed = (nxt < nb) & ((nxt - np.arange(nb)) * block_dr <= max_gap)
        use = past & (lab | closed)
        if valid is not None:
            use &= np.asarray(valid[p], dtype=bool)
        out.append(use)
    return out


def _margin_and_range(z_ref: float, s, delta: float, tau: float, d6, q, bary, bary_c, q_c, grad_op, has):
    r"""Margin :math:`h_k` and range :math:`r_k` of one sample from its local depths d6 (own 3, crest 3).

    The margin is the soft maximum of the crest's height above the line of sight and the face's
    own, :math:`-\delta \cos\varphi_k` (see the module docstring).
    """
    dk, dc = d6[:3], d6[3:]
    P = jnp.concatenate([q, (z_ref - bary @ dk)[None]])
    Pc = jnp.concatenate([q_c, (z_ref - bary_c @ dc)[None]])
    alpha = jnp.arctan2(P[2] - s[2], jnp.linalg.norm(P[:2] - s[:2]))
    alpha_c = jnp.arctan2(Pc[2] - s[2], jnp.linalg.norm(Pc[:2] - s[:2]))
    crest = jnp.where(has, jnp.linalg.norm(Pc - s) * jnp.sin(alpha_c - alpha), -H_OPEN)
    r = jnp.linalg.norm(P - s)
    g = grad_op @ jnp.stack([dk[1] - dk[0], dk[2] - dk[0]])
    n = jnp.concatenate([g, jnp.ones(1)])
    face = delta * jnp.dot(n / jnp.linalg.norm(n), (P - s) / r)  # :math:`-\delta \cos\varphi`
    h = tau * jnp.logaddexp(crest / tau, face / tau)
    return jnp.stack([h, r])


def _soft_min(h, r, ok, tau: float, width: float, offset: float, nb: int):
    """Block margins m (nb,), sample weights den (nb,), and per sample its two blocks, their weights and dm/dh, dm/dr."""
    x = r / width - offset
    i0 = jax.lax.stop_gradient(jnp.floor(x)).astype(jnp.int32)
    f = x - i0
    blk = jnp.stack([i0, i0 + 1], axis=1)  # (K, 2)
    pi = jnp.stack([1 - f, f], axis=1) * ok[:, None]
    dpi = jnp.stack([-jnp.ones_like(f), jnp.ones_like(f)], axis=1) / width * ok[:, None]
    lg = -h / tau
    inb = (blk >= 0) & (blk < nb) & (pi > 0)
    bc = jnp.clip(blk, 0, nb - 1)
    top = jnp.full(nb, -jnp.inf).at[bc].max(jnp.where(inb, lg[:, None], -jnp.inf))
    top = jnp.where(jnp.isfinite(top), top, 0.0)
    e = jnp.where(inb, jnp.exp(lg[:, None] - top[bc]), 0.0)  # (K, 2), stabilised per block
    num = jnp.zeros(nb).at[bc].add(jnp.where(inb, pi * e, 0.0))
    den = jnp.zeros(nb).at[bc].add(jnp.where(inb, pi, 0.0))
    has = num > 0
    m = jnp.where(has, -tau * (top + jnp.log(jnp.where(has, num, 1.0)) - jnp.log(jnp.where(has, den, 1.0))), 0.0)
    num_k, den_k = num[bc], den[bc]
    dm_dh = jnp.where(inb, pi * e / jnp.where(num_k > 0, num_k, 1.0), 0.0)
    dm_dr = jnp.where(inb, -tau * dpi * (e / jnp.where(num_k > 0, num_k, 1.0) - 1.0 / jnp.where(den_k > 0, den_k, 1.0)),
                      0.0)
    return m, den, blk, dm_dh, dm_dr


@cache
def _batched(mesh: M.Mesh, side: int, u_key: tuple, margin: float, z_water: float, width: float, offset: float,
             nb: int, tau: float, rows: bool):
    r"""Jitted (d, poses) -> margins (P, nb), weights (P, nb), and with `rows` the local rows: blocks (P, K, 2), coefficients (P, K, 2, 6), columns (P, K, 6)."""
    u = np.asarray(u_key)
    idx = np.searchsorted(u, u - margin, side="right") - 1  # as `geometry.horizon_visibility`
    has0 = jnp.asarray(idx >= 0)
    idx0 = jnp.asarray(np.maximum(idx, 0))
    tris = jnp.asarray(mesh.triangles)
    grad_ops = jnp.asarray(mesh._grad_ops)

    def one(d, pose):
        s = G.pose_frame(pose)[0]
        q = G.trace(mesh, d, pose, side, jnp.asarray(u))
        tri, bary, inside = M.locate(mesh, q)
        verts = tris[tri]
        P = jnp.concatenate([q, (mesh.z_ref - jnp.sum(d[verts] * bary, axis=-1))[:, None]], axis=-1)
        j = G._running_max(G.elevation(P, s, inside))[1][idx0]
        has = has0 & inside[j]  # an off-mesh crest is no crest
        ok = inside & (P[:, 2] <= z_water)
        cols = jnp.concatenate([verts, verts[j]], axis=1)
        local = d[cols]

        def f(d6, q, bary, bary_c, q_c, grad_op, has):
            return _margin_and_range(mesh.z_ref, s, margin, tau, d6, q, bary, bary_c, q_c, grad_op, has)

        args = (local, q, bary, bary[j], q[j], grad_ops[tri], has)
        hr = jax.vmap(f)(*args)
        m, den, blk, dm_dh, dm_dr = _soft_min(hr[:, 0], hr[:, 1], ok, tau, width, offset, nb)
        if not rows:
            return m, den
        jac = jax.vmap(jax.jacfwd(f))(*args)  # (K, 2, 6)
        coef = dm_dh[:, :, None] * jac[:, None, 0, :] + dm_dr[:, :, None] * jac[:, None, 1, :]
        return m, den, blk, coef, cols

    return jax.jit(jax.vmap(one, in_axes=(None, 0)))


def label_prob(m, p0, hz: Horizon):
    r"""(:math:`p`, :math:`\partial p / \partial m`) of a lit label at margins m with lit rate p0 without seabed."""
    z = np.asarray(m) / hz.sigma
    a = (1 - p0) * (1 - hz.eps)
    return p0 + a * norm.cdf(-z), -a * norm.pdf(z) / hz.sigma


def margins(mesh: M.Mesh, d, poses, side: int, theta: Theta, u, dr: float, n_blocks: int, hz: Horizon):
    """Block margins (n, n_blocks) and sample weights (n, n_blocks) of one side of every pose in poses (n, 6)."""
    B = theta.block
    f = _batched(mesh, side, tuple(np.asarray(u).tolist()), float(theta.shadow_margin), float(theta.z_water),
                 B * dr, (B - 1) / (2 * B), n_blocks, hz.tau, False)
    m, den = f(jnp.asarray(d, dtype=float), jnp.asarray(poses))
    return np.asarray(m), np.asarray(den)


def horizon_term(mesh: M.Mesh, d, poses: list[np.ndarray], labels: list[np.ndarray], theta: Theta, u, dr: float,
                 hz: Horizon, rows: bool = True, chunk: int = 32):
    r"""Over all passes: :math:`\sum \log p(\ell_j \mid d)` over the scored blocks, its gradient in d (n,), and the GGN rows.

    Returns (value, grad, A, W) with A (m, n) sparse, one row :math:`\partial m_j / \partial d` per
    scored block, and W (m,) the Fisher weights, so the term's GGN is :math:`A^\top W A`. Without
    `rows`, only the value (grad, A, W None).
    """
    if hz.use is None:
        raise ValueError("Horizon.use is not set: build it with scored_blocks")
    n = mesh.n_vertices
    B = theta.block
    d = jnp.asarray(d, dtype=float)
    u_key = tuple(np.asarray(u).tolist())
    total = 0.0
    grad = np.zeros(n) if rows else None
    rr, cc, vv, ww = [], [], [], []
    m_rows = 0
    for P, lab, use in zip(poses, labels, hz.use, strict=True):
        nb = lab.shape[-1]
        for side in (0, 1):
            p0 = noise_lit_rate(theta, side, nb)
            f = _batched(mesh, side, u_key, float(theta.shadow_margin), float(theta.z_water), B * dr,
                         (B - 1) / (2 * B), nb, hz.tau, rows)
            for i in range(0, len(P), chunk):
                out = [np.asarray(a) for a in f(d, jnp.asarray(P[i:i + chunk]))]
                m, den = out[0], out[1]
                lit = np.asarray(lab[i:i + chunk, side], dtype=bool)
                sc = np.asarray(use[i:i + chunk, side], dtype=bool) & (den >= hz.den_min)
                if not sc.any():
                    continue
                p, dp = label_prob(m, p0[None, :], hz)
                p = np.clip(p, 1e-12, 1 - 1e-12)
                lp = np.where(lit, np.log(p), np.log1p(-p))
                total += float(lp[sc].sum())
                if not rows:
                    continue
                g = np.where(lit, dp / p, -dp / (1 - p))  # :math:`\partial \log p(\ell) / \partial m`
                W = dp**2 / (p * (1 - p))
                blk, coef, cols = out[2], out[3], out[4]
                for t in range(m.shape[0]):
                    s_t = sc[t]
                    if not s_t.any():
                        continue
                    b = blk[t]  # (K, 2)
                    inb = (b >= 0) & (b < nb)
                    bc = np.clip(b, 0, nb - 1)
                    keep = inb & s_t[bc]
                    kk, aa = np.nonzero(keep)
                    c = coef[t][kk, aa]  # (e, 6)
                    rows_b = bc[kk, aa]
                    vid = cols[t][kk]  # (e, 6)
                    np.add.at(grad, vid, (g[t][rows_b])[:, None] * c)
                    # one GGN row per scored block: sum duplicates, then drop entries tiny against the row's largest
                    order = np.flatnonzero(s_t)
                    row_of = np.full(nb, -1)
                    row_of[order] = np.arange(len(order))
                    Rc = sp.coo_matrix((c.ravel(), (np.repeat(row_of[rows_b], 6), vid.ravel())),
                                       shape=(len(order), n)).tocsr().tocoo()  # duplicates summed
                    big = np.zeros(len(order))
                    np.maximum.at(big, Rc.row, np.abs(Rc.data))
                    k = np.abs(Rc.data) > hz.row_tol * big[Rc.row]
                    rr.append(Rc.row[k] + m_rows)
                    cc.append(Rc.col[k])
                    vv.append(Rc.data[k])
                    ww.append(W[t][order])
                    m_rows += len(order)
    if not rows:
        return total, None, None, None
    if m_rows == 0:
        return total, grad, sp.csr_matrix((0, n)), np.zeros(0)
    A = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(m_rows, n))
    return total, grad, A, np.concatenate(ww)

