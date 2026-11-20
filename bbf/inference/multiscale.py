r"""Coarse-to-fine inference in range: smooth the depth likelihood before resolving bin-scale detail.

At bin scale the depth likelihood is rough (range timing and speckle), so Gauss--Newton from
the first-echo plane stops at the nearest local optimum. Averaging the intensities over
:math:`k` bins smooths it at the scale of the start's error. Each level runs the outer loop
(`bbf.inference.loop.infer`) on coarse data from the previous level's depths, down to
:math:`k = 1`.

For odd :math:`k`, coarse bin :math:`j` is the mean of the fine bins centred on
:math:`j k \Delta r`, so its centre matches the renderer's bins of width :math:`k \Delta r`. The
mean of :math:`k` gamma bins of shape :math:`L` and similar means is approximately
:math:`\mathrm{Gam}(kL)`; a spike in one of them, with probability :math:`1 - (1 - \epsilon)^k`,
raises the mean by about the spike over :math:`k`, so :math:`\eta` and :math:`\zeta` scale by
:math:`1/k`. Labels take one block per coarse bin.
"""
import dataclasses

import numpy as np

from bbf import mesh as M
from bbf.inference.loop import infer
from bbf.io import Pass
from bbf.ping.render import DU, u_grid
from bbf.ping.theta import Theta
from bbf.prior import Prior


def coarsen_pass(ps: Pass, k: int) -> Pass:
    r"""Pass with intensities averaged over :math:`k` bins (k odd), bin spacing :math:`k \Delta r`."""
    if k == 1:
        return ps
    if k % 2 == 0:
        raise ValueError("k must be odd so coarse bins are centred on the fine ones")
    h = (k - 1) // 2
    B = ps.n_bins
    n = (B - 1 - h) // k + 1
    c = np.cumsum(np.concatenate([np.zeros(ps.y.shape[:-1] + (1,)), ps.y], axis=-1), axis=-1)
    lo = np.clip(np.arange(n) * k - h, 0, B)
    hi = np.clip(np.arange(n) * k + h + 1, 0, B)
    y = (c[..., hi] - c[..., lo]) / (hi - lo)
    return dataclasses.replace(ps, y=y, dr=k * ps.dr, name=f"{ps.name}/k{k}")


def coarsen_theta(theta: Theta, k: int) -> Theta:
    """Parameters of the k-bin means: shape kL, spike chance per k bins, spike levels over k, one label block per bin."""
    if k == 1:
        return theta
    eps = 1.0 - (1.0 - np.asarray(theta.eps)) ** k
    return dataclasses.replace(theta, L=k * theta.L, eps=eps, eta=np.asarray(theta.eta) / k,
                               zeta=np.asarray(theta.zeta) / k, block=1)


def infer_multiscale(passes: list[Pass], mesh: M.Mesh, prior: Prior, theta: Theta, d0, beta0,
                     levels=((27, 2, 8), (9, 2, 8), (3, 2, 8)), du: float = DU, n_draws: int = 4000,
                     calib_every: int = 50, chunk: int = 32, callback=None, log=print) -> dict:
    r"""Run `infer` at each level (k, rounds, inner GN steps), coarse to fine; returns the last level's result.

    The default stops at :math:`k = 3`: on the Örebro data the seabed speckle stays correlated
    over 2--3 bins of 1.32 cm (seabed texture along range), so finer bins would count the same
    information several times as if independent.

    The curve is sampled at :math:`\max(du, k\Delta r / 3)`, so coarse levels are also cheap.
    `callback(level_index, k, result)` runs after each level.
    """
    d, beta = np.asarray(d0, dtype=float), np.asarray(beta0, dtype=float)
    out, results = None, []
    for li, (k, rounds, inner) in enumerate(levels):
        cp = [coarsen_pass(p, k) for p in passes]
        th = coarsen_theta(theta, k)
        u = u_grid(cp[0].n_bins, cp[0].dr, du=max(du, cp[0].dr / 3))
        log(f"level k = {k}: {cp[0].n_bins} bins of {cp[0].dr:.3f} m, {len(u)} samples per side")
        out = infer(cp, mesh, prior, th, d, beta, u, rounds=rounds, inner=inner, n_draws=n_draws,
                    calib_every=calib_every, chunk=chunk, log=log)
        d, beta = out["d"], out["beta"]
        results.append({"k": k, "history": out["history"]})
        if callback is not None:
            callback(li, k, out)
    assert out is not None
    out["levels"] = results
    return out
