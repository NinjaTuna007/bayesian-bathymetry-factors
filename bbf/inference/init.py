r"""Starting point of inference: the mesh over the passes and the first echoes.

Each ping-side's first echo is the nearest seabed in range on that side. Its slant range
:math:`r_0`, where the intensity first jumps far above the water column (`first_echo`), bounds
the depth below the sonar; the later side's puts a seabed point roughly below it at
:math:`(x_t, y_t, z_t - r_0)`. A robust plane through those points gives the trend
coefficients :math:`\beta_0` (and the plane depths :math:`X \beta_0`), and the GMRF conditional mean
given the points gives starting depths that follow the relief under the tracks
(`depths_from_points`). Where a ping's two sides echo at different ranges, the nearer side's
seabed rises towards the sonar (a bank); `bank_points` adds points for it.
"""
import warnings

import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cho_factor

from bbf import mesh as M
from bbf.io import Pass
from bbf.ping.factor import background, bin_ranges, log_gain
from bbf.ping.theta import Theta
from bbf.prior import Prior


def swath_box(poses, max_range: float) -> tuple[float, float, float, float]:
    """Axis-aligned box (xmin, xmax, ymin, ymax) holding every ping's swath, from the poses (n, 6)."""
    poses = np.asarray(poses)
    yaw = poses[:, 5]
    sx, sy = max_range * np.cos(yaw), -max_range * np.sin(yaw)  # starboard is :math:`(\cos\psi, -\sin\psi)`
    ex = np.concatenate([poses[:, 0] + sx, poses[:, 0] - sx])
    ey = np.concatenate([poses[:, 1] + sy, poses[:, 1] - sy])
    return float(ex.min()), float(ex.max()), float(ey.min()), float(ey.max())


def mesh_for_passes(passes: list[Pass], h: float = 0.5, buffer: float = 10.0) -> M.Mesh:
    r"""Mesh over the union of the passes' swaths plus `buffer`; :math:`z_\mathrm{ref}` is the median sonar height."""
    boxes = np.array([swath_box(p.poses, p.n_bins * p.dr) for p in passes])
    z_ref = float(np.median(np.concatenate([p.poses[:, 2] for p in passes])))
    return M.build(boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(), boxes[:, 3].max(), h=h,
                   buffer=buffer, z_ref=z_ref)


def _onset(mt, start: int, run: int, w: int, jump: float, leaked=None):
    """First block index of the jump in log block means mt at or beyond block `start`, or None (see `first_echo`).

    leaked: optional boolean per block; a candidate onset at such a block is passed over.
    """
    for j in range(max(start, w), len(mt) - run + 1):
        base = np.median(mt[j - w:j])
        if np.median(mt[j:j + run]) > base + np.log(jump):
            k = j + int(np.argmax(mt[j:j + run] > base + 0.5 * np.log(jump)))
            if leaked is None or not leaked[k]:
                return k
    return None


def first_echo(ps: Pass, theta: Theta, min_range: float = 3.0, run: int = 5, jump: float = 20.0,
               window: float = 2.0, crosstalk_db: float = -34.0, leak: float = 6.0) -> np.ndarray:
    r"""Slant range (n, 2) of each ping-side's first echo; NaN where none is found.

    The seabed's first echo is a jump of orders of magnitude over the water column, while
    transmit ringing decays smoothly and fish are isolated. With :math:`m_j` the log block mean
    of :math:`y / g N` (blocks of `Theta.block` bins), the jump is found at the first block
    beyond `min_range` metres where the median of the next `run` blocks exceeds the median
    of the preceding `window` metres by the factor `jump`; medians keep one dark speckle
    block or one fish from deciding. The echo is the first block of that run above
    :math:`\sqrt{\text{jump}}`, the half-way level in log, and its range is the block's centre.

    Cross-side leakage: each channel carries a copy of the other's echo envelope some
    :math:`35` dB down (on pass A of the 2026-07-07 line 1, port's water column under starboard's
    bank is :math:`-38` dB of starboard's, correlated with it at 0.73 at any lag, i.e. the envelope,
    not the speckle; port fired with starboard on 70% of pings). Block means are first cleaned
    of `crosstalk_db` times the other side's (floored at 2% of their value), and a candidate onset
    is still passed over when the other side has already echoed and its raw log level (no
    gain, so the channels compare) over the same `run` blocks is more than `leak` nats higher.
    On A this leaves port's floor ambiguous on some pings (8% none found), which the
    first-return factor's outlier component absorbs. Port's sensor-fixed lines (strongest at
    2.2--3.8 m on this sonar) lie inside `min_range`.
    """
    blk = theta.block * ps.dr
    first_block = int(np.ceil(min_range / blk))
    w = max(round(window / blk), 1)
    r = np.asarray(bin_ranges(ps.n_bins, ps.dr))
    n_blocks = ps.n_bins // theta.block
    nb = n_blocks * theta.block
    lin = ps.y[:, :, :nb].reshape(ps.n_pings, 2, n_blocks, theta.block).mean(axis=3)
    lin = np.maximum(lin - 10 ** (crosstalk_db / 10) * lin[:, ::-1], 0.02 * lin)  # the other side's leak removed
    raw = np.log(lin + 1e-300)
    # raw level of the next `run` blocks, per block
    pad = np.concatenate([raw, np.repeat(raw[..., -1:], run - 1, axis=-1)], axis=-1)
    level = np.median(np.lib.stride_tricks.sliding_window_view(pad, run, axis=-1), axis=-1)
    m = np.empty_like(raw)
    for c in (0, 1):
        st = theta.at(c)
        g = np.exp(np.asarray(log_gain(r, st.gain_db)))[:nb]
        # gain and background are smooth over a block; the background may follow each ping's power
        noise = np.asarray(background(theta, c, r[:nb], ps.log_power()[:, None]))
        gb = (g * noise).reshape(-1, n_blocks, theta.block).mean(axis=2)
        m[:, c] = np.log(lin[:, c] / gb + 1e-12)
    k = np.full((ps.n_pings, 2), -1)
    for t in range(ps.n_pings):
        plain = [_onset(m[t, c], first_block, run, w, jump) for c in (0, 1)]
        for c in (0, 1):
            other = plain[1 - c]
            leaked = None
            if other is not None:
                leaked = (np.arange(n_blocks) >= other) & (level[t, 1 - c] - level[t, c] > leak)
            j = _onset(m[t, c], first_block, run, w, jump, leaked)
            k[t, c] = -1 if j is None else j
    out = np.full((ps.n_pings, 2), np.nan)
    ok = k >= 0
    out[ok] = (k[ok] + 0.5) * blk
    return out


def far_edge(ps: Pass, theta: Theta, r0: np.ndarray, drop: float = 3.0, recover: float = 1.0,
             near: tuple[float, float] = (2.0, 8.0), smooth: float = 2.0, min_tail: float = 5.0) -> np.ndarray:
    r"""Slant range (n, 2) beyond which each ping-side sees no more seabed; NaN where it sees seabed to the end.

    With :math:`e_b = \log y_b - \log g_c(r_b) - o_t - \log S^\flat_b` the log intensity corrected for
    the gain, the transmit power and the unit intensity of a level seabed at the depth :math:`D_t` below
    the sonar (the later first echo),

    .. math::

        S^\flat(r) = B_c(\beta)\,\frac{\sin^2\beta}{\cos\beta}\,\frac{e^{-4\alpha r}}{r^3}, \qquad \sin\beta = D_t / r

    (`bbf.ping.geometry.echo` on a level seabed, :math:`\cos\beta` from the footprint per range),
    median-filtered over `smooth` m, so that the fading with grazing angle is not taken for an edge, and :math:`\bar e` its median over the `near` window past the
    first echo :math:`r_0` (the seabed just beyond the onset), the edge is the first bin past that
    window where :math:`e_b < \bar e - \text{drop}` and the intensity never comes back within
    `recover` of that level for the rest of the ping (at least `min_tail` m of it):

    .. math::

        r_e = \min\Bigl\{ r_b : \max_{b' \ge b} e_{b'} < \bar e - \text{drop} + \text{recover} \Bigr\}.

    A shadow behind a crest ends where the seabed comes back; the far edge does not. On the
    2026-07-07 Örebro passes starboard's edge is the shore: edges of different passes fall on
    one contour of the ground. Beyond it the intensity is still some 200 times the water
    column's (reverberation), so labels against the measured floor call it lit seabed.
    """
    from scipy.ndimage import median_filter

    r = np.asarray(bin_ranges(ps.n_bins, ps.dr))
    w = max(round(smooth / ps.dr), 1)
    out = np.full((ps.n_pings, 2), np.nan)
    D = np.max(np.where(np.isfinite(r0), r0, -np.inf), axis=1)
    sin_b = np.clip(D[:, None] / np.maximum(r[None], 1e-9), 1e-3, 1.0 - 1e-9)
    beta = np.arcsin(sin_b)
    for c in (0, 1):
        log_B = -4 * np.log(2) * (beta - np.asarray(theta.beta0)[c]) ** 2 / np.asarray(theta.omega)[c] ** 2
        if theta.beam_knots is not None:
            log_B = log_B + np.interp(beta, np.asarray(theta.beam_knots), np.asarray(theta.beam_log)[c])
        log_flat = log_B + 2 * np.log(sin_b) - 0.5 * np.log1p(-sin_b**2) - 4 * theta.alpha * r[None] - 3 * np.log(r[None])
        with np.errstate(divide="ignore"):
            e = (np.log(np.maximum(ps.y[:, c], 1e-300)) - np.asarray(log_gain(r, np.asarray(theta.gain_db)[c]))[None]
                 - ps.log_power()[:, None] - log_flat)
        e = median_filter(e, size=(1, w), mode="nearest")
        tail = np.maximum.accumulate(e[:, ::-1], axis=1)[:, ::-1]  # the most the intensity comes back beyond each bin
        for t in range(ps.n_pings):
            if not np.isfinite(r0[t, c]):
                continue
            j0, j1 = np.searchsorted(r, r0[t, c] + near[0]), np.searchsorted(r, r0[t, c] + near[1])
            if j1 >= ps.n_bins:
                continue
            ref = np.median(e[t, j0:j1])
            ok = (tail[t, j1:] < ref - drop + recover) & (e[t, j1:] < ref - drop) & (r[j1:] <= r[-1] - min_tail)
            if ok.any():
                out[t, c] = r[j1 + int(np.argmax(ok))]
    return out


def far_edge_points(ps: Pass, edge: np.ndarray, rise: float = 0.0, window: int = 9) -> np.ndarray:
    r"""World points (m, 3) where the seabed leaves the water, from the far edges (n, 2) of `far_edge`.

    Beyond its far edge a ping-side sees no seabed again, as where a bank reaches the water
    surface. With the surface `rise` m above the sonar (a hull-mounted sonar: about its draft),
    the seabed at slant range :math:`r_e` on that side is at the surface, a horizontal distance
    :math:`\sqrt{r_e^2 - \text{rise}^2}` from the sonar across the track. Edges are first
    median-filtered over `window` pings of the side, which drops single-ping detections.
    """
    from scipy.ndimage import median_filter

    yaw = ps.poses[:, 5]
    stb = np.column_stack([np.cos(yaw), -np.sin(yaw)])
    out = []
    for c, sgn in ((0, -1.0), (1, 1.0)):
        e = np.asarray(edge[:, c], dtype=float)
        ok = np.isfinite(e)
        if not ok.any():
            continue
        filled = np.where(ok, e, np.inf)
        sm = median_filter(filled, size=window, mode="nearest")  # inf where most of the window has no edge
        keep = ok & np.isfinite(sm) & (np.abs(e - sm) < 3.0)
        x = np.sqrt(np.maximum(sm[keep] ** 2 - rise**2, 0.0))
        xy = ps.poses[keep, :2] + sgn * x[:, None] * stb[keep]
        out.append(np.column_stack([xy, ps.poses[keep, 2] + rise]))
    return np.concatenate(out) if out else np.zeros((0, 3))


def first_echo_points(ps: Pass, r0: np.ndarray) -> np.ndarray:
    r"""World seabed points (m, 3) below the sonar from the later first echo of each ping.

    Each side's first echo is the nearest seabed in its half of the ping plane, never farther
    than the seabed below the sonar; on a slope the up-slope side's comes from the bank. The
    later of the two is the tighter bound on the depth below the sonar, so it places the point.
    """
    fin = np.isfinite(r0)
    r = np.where(fin.any(axis=1), np.max(np.where(fin, r0, -np.inf), axis=1), np.nan)
    ok = np.isfinite(r)
    return np.column_stack([ps.poses[ok, 0], ps.poses[ok, 1], ps.poses[ok, 2] - r[ok]])


def bank_points(ps: Pass, r0: np.ndarray, min_gap: float = 1.0, step: float = 1.0, window: int = 9,
                top: float | None = None) -> np.ndarray:
    r"""World seabed points (m, 3) on the side whose first echo is the earlier one, so that a start honours both.

    With :math:`D` the later first echo of a ping (the depth below the sonar, `first_echo_points`)
    and :math:`r` the earlier one, the nearer side's seabed comes within :math:`r` of the sonar. The
    simplest surface that does is a plane rising from the point below the sonar at the angle
    :math:`\gamma` with :math:`\cos\gamma = r / D`, which touches the range :math:`r` at the horizontal distance
    :math:`r \sin\gamma` outboard. The points lie on that plane every `step` m out to the touching
    point, for the pings whose echoes differ by more than `min_gap` m. The echoes are first
    replaced by their running median over `window` pings per side: single detections are off by
    metres on some pings (a fish, the other channel's leak), and each would raise a bank of its own.
    Where along the swath the seabed really rises is not in the first echoes; the intensities have
    to move it from there.

    top: continue the plane beyond the touching point until it is `top` m below the sonar, for a
    bank that rises to the surface under a sonar near it. Stopped at the touching point, the start
    falls back to the prior mean behind it: a ridge, whose far side is in shadow where a bank
    that goes on up is still lit.
    """
    r0 = np.asarray(r0, dtype=float)
    h = window // 2
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # windows without a detection give NaN
        r0 = np.stack([np.nanmedian(r0[max(t - h, 0):t + h + 1], axis=0) for t in range(len(r0))])
    pts = []
    for pose, (rp, rs) in zip(ps.poses, r0):
        if not (np.isfinite(rp) and np.isfinite(rs)) or abs(rp - rs) <= min_gap:
            continue
        D, r, sign = max(rp, rs), min(rp, rs), (1.0 if rs < rp else -1.0)
        out = sign * np.array([np.cos(pose[5]), -np.sin(pose[5])])  # starboard is :math:`(\cos\psi, -\sin\psi)`
        cg = r / D
        sg = np.sqrt(1.0 - cg**2)
        reach = r * sg
        if top is not None:
            reach = max(reach, (D - top) * cg / sg)
        u = np.arange(step, reach, step)
        u = np.append(u, reach)
        pts.append(np.column_stack([pose[0] + u * out[0], pose[1] + u * out[1], pose[2] - D + u * sg / cg]))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def plane_fit(points: np.ndarray, mesh: M.Mesh, iters: int = 20, k: float = 1.345) -> np.ndarray:
    r"""Trend coefficients :math:`\beta` of the depth plane :math:`d = \beta_1 x + \beta_2 y + \beta_3` (local).

    Huber IRLS on the depths :math:`z_\mathrm{ref} - z` of world points (m, 3); the scale is the
    MAD of the residuals, so stray echoes (fish, ringing) are down-weighted.
    """
    q = mesh.to_local(points[:, :2])
    A = np.column_stack([q, np.ones(len(q))])
    depth = mesh.z_ref - points[:, 2]
    w = np.ones(len(q))
    beta = np.zeros(3)
    for _ in range(iters):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(A * sw[:, None], depth * sw, rcond=None)[0]
        res = depth - A @ beta
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        u = np.abs(res) / (k * scale)
        w = np.where(u <= 1, 1.0, 1.0 / u)
    return beta


def initial_depths(mesh: M.Mesh, beta: np.ndarray) -> np.ndarray:
    r"""Vertex depths :math:`d_0 = X \beta` of the plane."""
    v = mesh.vertices
    return beta[0] * v[:, 0] + beta[1] * v[:, 1] + beta[2]


def depths_from_points(mesh: M.Mesh, points: np.ndarray, prior: Prior, sigma_obs: float = 0.5) -> np.ndarray:
    r"""Mesh depths from world seabed points: vertex averages, conditional mean of the GMRF elsewhere.

    Each point counts as an observation of its nearest vertex's depth with standard deviation
    `sigma_obs`; the result maximises :math:`\log p(d \mid \beta) + \sum \log \mathcal{N}(\text{obs} \mid d, \sigma_\mathrm{obs}^2)`
    at the prior's trend mean.
    """
    q = mesh.to_local(points[:, :2])
    tri, bary, inside = (np.asarray(a) for a in M.locate(mesh, q))
    vert = mesh.triangles[tri, np.argmax(bary, axis=1)][inside]
    depth = (mesh.z_ref - points[:, 2])[inside]
    n = np.bincount(vert, minlength=mesh.n_vertices).astype(float)
    total = np.bincount(vert, depth, minlength=mesh.n_vertices)
    W = sp.diags(n / sigma_obs**2)
    mean_trend = prior.X @ prior.mu_beta
    rhs = prior.Lambda @ mean_trend + total / sigma_obs**2
    return cho_factor(sp.csc_matrix(prior.Lambda + W)).solve(rhs)
