r"""Ratio match: views of the same ground must agree in the ratio of measured to predicted intensity.

The seabed's reflectivity :math:`\rho` multiplies every echo from a location, whichever pass sees it,
and is unknown. Bins are grouped by where they land: each scored bin's footprint (`cells.footprint`,
from the render at the current depths) falls in a square ground cell. Within a cell, the bins of one
*visit*, one pass seeing the cell in one continuous stretch (same pass, no gap longer than `gap`
seconds, so a track that loops back over the cell is a second visit), sum to

.. math::

    Y_v = \sum_b y_b, \qquad \bar I_v = \sum_b \bar I_b, \qquad k_v = L\,\frac{(\sum_b \bar I_b)^2}{\sum_b \bar I_b^2},

with :math:`Y_v` close to :math:`\mathrm{Gam}(k_v, \text{mean } \rho\,\bar I_v)`, :math:`\bar I_b` predicted
without the reflectivity. All visits of a cell share :math:`\rho`; integrated out under
:math:`p(\rho) \propto 1/\rho`,

.. math::

    \log p(Y_{1..m}) = \log\Gamma(K) - \sum_v \log\Gamma(k_v) + \sum_v k_v \log\frac{k_v Y_v}{\bar I_v}
        - K \log \sum_v \frac{k_v Y_v}{\bar I_v} - \sum_v \log Y_v, \qquad K = \sum_v k_v,

which depends on the predictions only through their ratios: scaling every :math:`\bar I_v` of a cell
by the same factor leaves it unchanged, so the reflectivity cancels exactly. For two visits it is
the density of the log-ratio of two gamma variables (`cells.log_ratio_pdf`) up to a term in the data.

Mixed with the ping factor: the bins of cells seen in at least `min_visits` visits are scored here
only; every other scored bin stays in its ping factor, which assumes one reflectivity per
ping-side. No bin is scored twice. The assignment of bins to cells and visits is held fixed while
the depths are refined, like the labels, and recomputed each round.

Held for the round means held over all the Gauss--Newton steps of a round
(`bbf.inference.gauss_newton`): the Jacobians are taken again after every step, so the curvature
follows the depths, but the cells, the visits, the shapes below and the ping-sides' gains
(`bbf.inference.objective.ratio_gains`) stay those of the round's start.

The shapes are held too (`RatioAssignment.shape`, from `visit_shapes` at the round's start). Followed
through the predictions, they let the factor widen itself: uneven predictions within a visit lower
:math:`k_v`, and a lower shape forgives any mismatch between the passes (on the real data the beam
spline grew to 25 log units for a gain of :math:`10^7` in the log posterior). They also gave vertices a
gradient with no curvature behind it.

Model error: the speckle alone has two visits agree to :math:`1/\sqrt{k_v}`, a few percent for sums over
many bins, while the gain, the beam and the angular law are known to tenths of a log unit. With a
log-normal error of standard deviation :math:`s` (`RatioMatch.model_sd`) on each visit's prediction, the
log of :math:`Y_v / \bar I_v` has variance :math:`\psi'(k_v) + s^2 \approx 1/k_v + s^2`, and the factor keeps
its form with the shape

.. math::

    k_v \leftarrow \frac{k_v}{1 + k_v s^2},

which is at most :math:`1/s^2`.

Its curvature in :math:`\eta_v = \log\bar I_v` is the Fisher information
:math:`\mathrm{diag}(k) - k k^\top / K` of each cell (rank :math:`m - 1`: a common scale is free).
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.scipy.special import gammaln


@dataclass(frozen=True)
class RatioMatch:
    """Constants of the ratio match: cell side (m), gap (s) that ends a visit, visits a cell needs to be matched,
    and the standard deviation of the model error on a visit's log prediction (0: speckle alone)."""
    cell: float = 1.0
    gap: float = 5.0
    min_visits: int = 2
    model_sd: float = 0.0


@dataclass(frozen=True, eq=False)
class RatioAssignment:
    """Bins to visits of matched cells, fixed for a round.

    unit: per pass (n, 2, B) int, the visit of each bin (-1: not in a matched cell); unit_cell (U,):
    the matched cell of each visit; Y (U,): measured sums; L: the bins' gamma shape; shape (U,): the
    visits' shapes :math:`k_v`, held for the round (`visit_shapes`), or None to follow the predictions.
    """
    unit: list[np.ndarray]
    unit_cell: np.ndarray
    n_cells: int
    Y: np.ndarray
    L: float
    shape: np.ndarray | None = None

    @property
    def n_units(self) -> int:
        return len(self.unit_cell)

    def matched(self) -> list[np.ndarray]:
        """Per pass (n, 2, B): bins scored by the ratio match (and so not by the ping factor)."""
        return [u >= 0 for u in self.unit]


def assign(footprints: list[np.ndarray], times: list[np.ndarray], ys: list[np.ndarray], uses: list[np.ndarray],
           L: float, rm: RatioMatch) -> RatioAssignment:
    """Cells and visits of the scored bins, from footprints per pass (n, 2, B, 2), ping times (n,), y and use (n, 2, B)."""
    rec_pass, rec_t, rec_key, rec_flat, rec_y = [], [], [], [], []
    for p, (q, t, y, use) in enumerate(zip(footprints, times, ys, uses, strict=True)):
        ok = np.asarray(use, bool) & np.isfinite(q).all(axis=-1) & (np.asarray(y) > 0)
        idx = np.flatnonzero(ok.ravel())
        key = np.floor(q.reshape(-1, 2)[idx] / rm.cell).astype(np.int64)
        rec_pass.append(np.full(len(idx), p))
        rec_t.append(np.repeat(np.asarray(t, float), ok.shape[1] * ok.shape[2])[idx])
        rec_key.append(key)
        rec_flat.append(idx)
        rec_y.append(np.asarray(y, float).ravel()[idx])
    rp, rt, rk, rf, ry = (np.concatenate(a) for a in (rec_pass, rec_t, rec_key, rec_flat, rec_y))
    units = [np.full(np.asarray(u).shape, -1, dtype=np.int64) for u in uses]
    if len(rp) == 0:
        return RatioAssignment(units, np.zeros(0, np.int64), 0, np.zeros(0), L)
    _, cell = np.unique(rk, axis=0, return_inverse=True)
    cell = cell.ravel()
    order = np.lexsort((rt, rp, cell))
    c, p, t = cell[order], rp[order], rt[order]
    new = np.ones(len(order), bool)
    new[1:] = (c[1:] != c[:-1]) | (p[1:] != p[:-1]) | (t[1:] - t[:-1] > rm.gap)
    visit = np.cumsum(new) - 1  # visit index of each record, in sorted order
    visit_cell = c[new]
    n_vis = np.bincount(visit_cell)
    keep_cell = n_vis >= rm.min_visits
    keep_visit = keep_cell[visit_cell]
    vid = np.full(len(visit_cell), -1)
    vid[keep_visit] = np.arange(keep_visit.sum())
    cid = np.full(len(keep_cell), -1)
    cid[keep_cell] = np.arange(keep_cell.sum())
    unit_of_rec = np.empty(len(order), np.int64)
    unit_of_rec[order] = vid[visit]
    for pp in range(len(units)):
        m = rp == pp
        flat = units[pp].reshape(-1)
        flat[rf[m]] = unit_of_rec[m]
    U = int(keep_visit.sum())
    Y = np.zeros(U)
    has = unit_of_rec >= 0
    np.add.at(Y, unit_of_rec[has], ry[has])
    return RatioAssignment(units, cid[visit_cell[keep_visit]], int(keep_cell.sum()), Y, L)


def visit_shapes(Ibar: list, a: RatioAssignment, model_sd: float = 0.0) -> np.ndarray:
    r"""Shapes (U,) of the visits at the predicted means Ibar (per pass (n, 2, B)): :math:`k_v = L (\sum_b \bar I_b)^2 / \sum_b \bar I_b^2`,
    then :math:`k_v / (1 + k_v s^2)` for a model error of standard deviation :math:`s` = model_sd."""
    unit = np.concatenate([np.asarray(u).ravel() for u in a.unit])
    I = np.concatenate([np.asarray(x, dtype=float).ravel() for x in Ibar])
    on = unit >= 0
    s1 = np.bincount(unit[on], I[on], a.n_units)
    s2 = np.bincount(unit[on], I[on] ** 2, a.n_units)
    k = a.L * s1**2 / np.maximum(s2, 1e-300)
    return k / (1.0 + k * model_sd**2)


def log_ratio_match(Ibar: list, a: RatioAssignment):
    r"""Total :math:`\sum_\text{cells} \log p(Y_{1..m})` from predicted means without reflectivity, per pass (n, 2, B); JAX."""
    if a.n_units == 0:
        return 0.0
    unit = jnp.concatenate([jnp.asarray(u).ravel() for u in a.unit])
    I = jnp.concatenate([jnp.ravel(x) for x in Ibar])
    on = unit >= 0
    seg = jnp.where(on, unit, 0)
    s1 = jax.ops.segment_sum(jnp.where(on, I, 0.0), seg, a.n_units)
    s2 = jax.ops.segment_sum(jnp.where(on, I**2, 0.0), seg, a.n_units)
    # the shape of each sum: held for the round when given, else it moves with the predictions
    k = a.L * s1**2 / s2 if a.shape is None else jnp.asarray(a.shape)
    Y = jnp.asarray(a.Y)
    cell = jnp.asarray(a.unit_cell)
    K = jax.ops.segment_sum(k, cell, a.n_cells)
    z = k * Y / s1
    total = (jnp.sum(gammaln(K)) - jnp.sum(gammaln(k)) + jnp.sum(k * jnp.log(z))
             - jnp.sum(K * jnp.log(jax.ops.segment_sum(z, cell, a.n_cells))) - jnp.sum(jnp.log(Y)))
    return total


def fisher_rows(Ibar: list, a: RatioAssignment):
    r"""Sparse rows (A, B) over all bins (passes ravelled in order), in the bins' predicted means :math:`\bar I_b`.

    The curvature in any variables :math:`\theta` is :math:`D^\top (A^\top A - B^\top B) D` with
    :math:`D = \partial \bar I / \partial \theta` (bins by variables): row v of A is
    :math:`\sqrt{k_v}\, \partial \eta_v / \partial \bar I`, row c of B is
    :math:`\sum_{v \in c} k_v\, \partial \eta_v / \partial \bar I \,/\, \sqrt{K_c}`, with
    :math:`\eta_v = \log \bar I_v` and :math:`\partial \eta_v / \partial \bar I_b = 1 / \bar I_v` for its bins.
    Ibar: predicted means per pass (n, 2, B).
    """
    unit = np.concatenate([np.asarray(u).ravel() for u in a.unit])
    I = np.concatenate([np.asarray(x).ravel() for x in Ibar])
    on = np.flatnonzero(unit >= 0)
    u = unit[on]
    s1 = np.bincount(u, I[on], a.n_units)
    s2 = np.bincount(u, I[on] ** 2, a.n_units)
    k = a.L * s1**2 / s2 if a.shape is None else np.asarray(a.shape, dtype=float)
    K = np.bincount(a.unit_cell, k, a.n_cells)
    coef = 1.0 / s1[u]  # d eta_v / d I_b
    n = len(unit)
    A = sp.csr_matrix((np.sqrt(k[u]) * coef, (u, on)), shape=(a.n_units, n))
    c = a.unit_cell[u]
    B = sp.csr_matrix((k[u] * coef / np.sqrt(K[c]), (c, on)), shape=(a.n_cells, n))
    return A, B
