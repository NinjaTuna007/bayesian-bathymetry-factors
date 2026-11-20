r"""Rendering one side of a ping into range bins.

The unit intensity of bin :math:`b`, the predicted intensity per unit reflectivity before
gain and noise, is the echo integrated along the bin's part of the seabed curve,

.. math::

    S_b = \frac{1}{\Delta r} \int_{\mathcal{C}_b} a_c(q)\, V(q)\, \mathrm{d}q
        \approx \frac{1}{\Delta r} \sum_k a_k V_k \Delta q_k\, w_b(r_k),

with :math:`\Delta q_k` the horizontal curve length of sample :math:`k` and :math:`V_k` the
horizon visibility with diffraction of `geometry.horizon_visibility`, computed from the same samples. Bin :math:`b` is
centred at range :math:`b \Delta r`, and :math:`w_b` splits each sample linearly between
its two nearest bins, :math:`w_b(r) = \max(0, 1 - |r / \Delta r - b|)`, so the bins move
smoothly with the depths and range timing carries a gradient. The weights of a sample sum
to one, so the split conserves :math:`\sum_b S_b \Delta r`.
"""
import jax
import jax.numpy as jnp
import numpy as np

from bbf import mesh as M
from bbf.ping import geometry as G
from bbf.ping.theta import Theta

DU = 0.005  # default horizontal sample spacing along the curve (m)


def u_grid(n_bins: int, dr: float, du: float = DU) -> np.ndarray:
    """Horizontal outboard distances of the curve samples, out to the last bin's range."""
    return np.arange(du / 2, n_bins * dr, du)


def curve_length(q):
    r"""Horizontal curve length :math:`\Delta q_k` (K,) owned by each sample of a curve q (K, 2)."""
    seg = jnp.linalg.norm(jnp.diff(q, axis=0), axis=-1)
    pad = jnp.zeros(1, dtype=seg.dtype)
    return 0.5 * (jnp.concatenate([seg, pad]) + jnp.concatenate([pad, seg]))


def split_into_bins(r, w, n_bins: int, dr: float):
    r"""Sum weights w (K,) at ranges r (K,) into n_bins bins, each split linearly between its two nearest bins."""
    x = r / dr
    i0 = jnp.floor(x)
    f = x - i0
    i0 = i0.astype(jnp.int32)
    out = jnp.zeros(n_bins, dtype=w.dtype)
    out = out.at[i0].add((1 - f) * w, mode="drop")
    return out.at[i0 + 1].add(f * w, mode="drop")


def unit_intensity(mesh: M.Mesh, d, pose, side: int, theta: Theta, u, n_bins: int, dr: float):
    r"""Unit intensity :math:`S_b` (n_bins,) of one side, differentiable in d and the pose.

    u: horizontal sample distances (K,), ordered outboard, see `u_grid`. Shadowed samples
    are weighted by their diffracted visibility; samples off the mesh add nothing.
    """
    P, n, inside = G.samples(mesh, d, pose, side, u, smooth=theta.smooth_normals)
    a, r = G.echo(theta, pose, side, P, n)
    V = G.horizon_visibility(P, G.pose_frame(pose)[0], u, theta.shadow_margin, theta.wavelength)
    wet = inside & (P[:, 2] <= theta.z_water)  # seabed above the water surface returns nothing
    w = jnp.where(wet, a * V * curve_length(P[:, :2]), 0.0) / dr
    return split_into_bins(r, w, n_bins, dr)


def unit_intensities(mesh: M.Mesh, d, poses, theta: Theta, u, n_bins: int, dr: float):
    r"""Unit intensities (n, 2, n_bins) of both sides of every pose in poses (n, 6), mesh-local.

    Maps `unit_intensity` over the poses one at a time, so memory stays that of one ping.
    """
    poses = jnp.asarray(poses)
    sides = [jax.lax.map(lambda pose, side=side: unit_intensity(mesh, d, pose, side, theta, u, n_bins, dr), poses)
             for side in (0, 1)]
    return jnp.stack(sides, axis=1)
