r"""Coarse bins for inference coarse to fine in range.

At bin scale the depth likelihood is rough (range timing and speckle), so Gauss--Newton from
the start stops at the nearest local optimum. Averaging the intensities over :math:`k` bins
smooths it at the scale of the start's error; the outer loop (`bbf.inference.loop.infer`)
then runs level by level on coarse passes and theta (`scripts/infer.py`).

For odd :math:`k`, coarse bin :math:`j` is the mean of the fine bins centred on
:math:`j k \Delta r`, so its centre matches the renderer's bins of width :math:`k \Delta r`. The
mean of :math:`k` gamma bins of shape :math:`L` and similar means is approximately
:math:`\mathrm{Gam}(kL)`; a spike in one of them, with probability :math:`1 - (1 - \epsilon)^k`,
raises the mean by about the spike over :math:`k`, so :math:`\eta` and :math:`\zeta` scale by
:math:`1/k`. Labels take one block per coarse bin. The shape :math:`kL` assumes independent bins;
`coarsen_theta` takes the effective looks of a level instead when given.
"""
import dataclasses

import numpy as np

from bbf.io import Pass
from bbf.ping.theta import Theta


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


def coarsen_theta(theta: Theta, k: int, looks: float | None = None) -> Theta:
    r"""Parameters of the k-bin means: shape kL, spike chance per k bins, spike levels over k, one label block per bin.

    looks: the shape of the k-bin means to use instead of :math:`kL`. Neighbouring bins are
    correlated (the pulse spans several), so the mean of :math:`k` of them has fewer effective
    looks than :math:`k L`; and since :math:`\mathrm{Gam}(y \mid L, \mu)` depends on :math:`\mu` through
    :math:`-L (\log\mu + y / \mu)`, a shape scaled by :math:`\tau` also tempers the bin's weight by
    :math:`\tau`, which is how correlation between neighbouring pings can be discounted.
    """
    if k == 1:
        return theta if looks is None else dataclasses.replace(theta, L=looks)
    eps = 1.0 - (1.0 - np.asarray(theta.eps)) ** k
    return dataclasses.replace(theta, L=k * theta.L if looks is None else looks, eps=eps,
                               eta=np.asarray(theta.eta) / k, zeta=np.asarray(theta.zeta) / k, block=1)
