r"""Equilateral triangle lattice over an axis-aligned rectangle, and hat-function interpolation.

Vertex :math:`(r, c)` sits at

.. math::

    v_{rc} = \left(x_0 + \left(c + \tfrac{1}{2}(r \bmod 2)\right) h,\;
                   y_0 + \tfrac{\sqrt{3}}{2} r h\right):

rows are :math:`\frac{\sqrt{3}}{2} h` apart and odd rows are shifted right by
:math:`h/2`, so every edge has length :math:`h`. The vertex index is
:math:`r \cdot n_\mathrm{cols} + c`.

Between rows :math:`r` and :math:`r+1`, cell :math:`(r, c)` holds two triangles, with
:math:`s = r \bmod 2`:

.. math::

    \text{lower} = (v_{r,c},\; v_{r,c+1},\; v_{r+1,c+s}), \qquad
    \text{upper} = (v_{r+1,c+1+s},\; v_{r+1,c+s},\; v_{r,c+1}),

both counter-clockwise. Cells run over :math:`c = -1, \dots, n_\mathrm{cols} - 2`; a
triangle exists if its three vertices do, which gives the zigzag left and right edges of
the domain.

Coordinates are local: east and north in metres relative to `origin`, so the planar trend
stays well conditioned. Depth is positive down from the reference height `z_ref` (world
up axis), so the seabed point at :math:`q` is :math:`(q_1, q_2, z_\mathrm{ref} - d(q))`.
"""
import math
from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
import numpy as np

ROW = math.sqrt(3) / 2  # row spacing per unit :math:`h`


@dataclass(frozen=True)
class Mesh:
    h: float
    x0: float
    y0: float
    n_rows: int
    n_cols: int
    origin: tuple[float, float] = (0.0, 0.0)  # world (E, N) of local (0, 0)
    z_ref: float = 0.0  # world height that depths are measured down from

    @property
    def n_vertices(self) -> int:
        return self.n_rows * self.n_cols

    @cached_property
    def vertices(self) -> np.ndarray:
        """(n, 2) local (x, y) of each vertex."""
        r, c = np.divmod(np.arange(self.n_vertices), self.n_cols)
        return np.stack([self.x0 + (c + 0.5 * (r % 2)) * self.h, self.y0 + r * ROW * self.h], axis=1)

    @cached_property
    def _cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Triangles (m, 3) and lookup (n_rows-1, n_cols+1, 2) from (r, c+1, upper) to triangle, -1 if none."""
        R, C = self.n_rows, self.n_cols
        r, c = np.meshgrid(np.arange(R - 1), np.arange(-1, C - 1), indexing="ij")
        s = r % 2

        def v(rr, cc):
            ok = (cc >= 0) & (cc < C)
            return np.where(ok, rr * C + cc, -1)

        lower = np.stack([v(r, c), v(r, c + 1), v(r + 1, c + s)], axis=-1)
        upper = np.stack([v(r + 1, c + 1 + s), v(r + 1, c + s), v(r, c + 1)], axis=-1)
        cand = np.stack([lower, upper], axis=2)  # (R-1, C+1, 2, 3)
        valid = (cand >= 0).all(axis=-1)
        lookup = np.full(valid.shape, -1, dtype=np.int64)
        lookup[valid] = np.arange(valid.sum())
        return cand[valid], lookup

    @cached_property
    def triangles(self) -> np.ndarray:
        """(m, 3) vertex indices, counter-clockwise."""
        return self._cells[0]

    @cached_property
    def _grad_ops(self) -> np.ndarray:
        r"""(m, 2, 2) maps :math:`(d_1 - d_0,\, d_2 - d_0)` of a triangle to its depth gradient."""
        p = self.vertices[self.triangles]
        E = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=1)
        return np.linalg.inv(E)

    def to_local(self, en: np.ndarray) -> np.ndarray:
        r"""World :math:`(E, N)` to local :math:`(x, y)`."""
        return np.asarray(en, dtype=float) - np.asarray(self.origin)


def build(xmin: float, xmax: float, ymin: float, ymax: float, h: float, buffer: float = 0.0,
          origin: tuple[float, float] | None = None, z_ref: float = 0.0) -> Mesh:
    """Lattice whose triangles cover the world box (xmin, xmax, ymin, ymax) grown by `buffer`.

    origin defaults to the centre of the box; all mesh coordinates are relative to it.
    z_ref is the world height depths are measured down from, e.g. the water surface.
    """
    if h <= 0 or buffer < 0:
        raise ValueError("need h > 0 and buffer >= 0")
    if origin is None:
        origin = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    left, right = xmin - origin[0] - buffer, xmax - origin[0] + buffer
    bottom, top = ymin - origin[1] - buffer, ymax - origin[1] + buffer
    # Even rows start :math:`h/2` left of `left` so the zigzag left edge never cuts into the box;
    # the right edge's innermost points are the even-row ends, which reach `right`.
    x0 = left - h / 2
    n_cols = math.ceil((right - x0) / h) + 1
    n_rows = math.ceil((top - bottom) / (ROW * h)) + 1
    return Mesh(h=float(h), x0=x0, y0=bottom, n_rows=n_rows, n_cols=n_cols,
                origin=(float(origin[0]), float(origin[1])), z_ref=float(z_ref))


def locate(mesh: Mesh, q):
    """Triangle and barycentric weights of local points q (..., 2).

    Returns (tri, bary, inside): tri (...) indexes mesh.triangles (0 where outside),
    bary (..., 3) matches the triangle's vertex order, inside (...) is False off the mesh.
    Differentiable in q within a triangle. Points on the domain boundary count as inside.
    """
    q = jnp.asarray(q)
    h = mesh.h
    # skew coordinates: vertex :math:`(a, r)` at :math:`a (h, 0) + r (\tfrac{h}{2}, \tfrac{\sqrt{3}}{2} h)`
    bf = (q[..., 1] - mesh.y0) / (ROW * h)
    af = (q[..., 0] - mesh.x0) / h - bf / 2
    # Clamp to the nearest existing cell; points beyond it get negative weights below,
    # and points on the top or right edge land in the last cell instead of past it.
    # The small nudge sends points that round to just below a vertex or edge into the cell
    # above/right, which always holds a triangle at the left zigzag edge.
    r = jnp.clip(jnp.floor(bf + 1e-9).astype(jnp.int32), 0, mesh.n_rows - 2)
    half = jnp.floor_divide(r, 2)
    c = jnp.clip(jnp.floor(af + 1e-9).astype(jnp.int32) + half, -1, mesh.n_cols - 2)  # offset column
    fa, fb = af - (c - half), bf - r
    lookup = jnp.asarray(mesh._cells[1])[r, c + 1]  # (..., 2): lower, upper
    b_lo = jnp.stack([1 - fa - fb, fa, fb], axis=-1)
    b_up = jnp.stack([fa + fb - 1, 1 - fa, 1 - fb], axis=-1)
    # Prefer the triangle on q's side of the diagonal; fall back to the other one when the
    # preferred triangle is missing (zigzag edge) and q lies on the shared diagonal.
    use_up = (fa + fb >= 1) & (lookup[..., 1] >= 0) | (lookup[..., 0] < 0)
    tri = jnp.where(use_up, lookup[..., 1], lookup[..., 0])
    bary = jnp.where(use_up[..., None], b_up, b_lo)
    inside = (tri >= 0) & (bary.min(axis=-1) >= -1e-9)
    return jnp.where(inside, tri, 0), bary, inside


def interpolate(mesh: Mesh, d, q):
    r"""Depth :math:`d(q) = \sum_i d_i \psi_i(q)` at local points q (..., 2); NaN off the mesh."""
    tri, bary, inside = locate(mesh, q)
    dv = jnp.asarray(d)[jnp.asarray(mesh.triangles)[tri]]
    return jnp.where(inside, jnp.sum(dv * bary, axis=-1), jnp.nan)


def seabed_point(mesh: Mesh, d, q):
    r"""Seabed point :math:`P(q) = (q_1, q_2, z_\mathrm{ref} - d(q))` (..., 3), local x, y and world z; NaN off the mesh."""
    q = jnp.asarray(q)
    return jnp.concatenate([q, (mesh.z_ref - interpolate(mesh, d, q))[..., None]], axis=-1)


def face_gradient(mesh: Mesh, d):
    r"""(m, 2) gradient :math:`\nabla d` of the depth on each triangle (constant per face)."""
    dv = jnp.asarray(d)[jnp.asarray(mesh.triangles)]
    dd = jnp.stack([dv[:, 1] - dv[:, 0], dv[:, 2] - dv[:, 0]], axis=-1)
    return jnp.einsum("mij,mj->mi", jnp.asarray(mesh._grad_ops), dd)


def normal(mesh: Mesh, d, q):
    r"""Upward unit normal (..., 3) of the seabed :math:`z = z_\mathrm{ref} - d(q)` at local points q; NaN off the mesh.

    .. math::

        n = \frac{(\nabla d,\, 1)}{\sqrt{1 + \lVert \nabla d \rVert^2}},

    constant over each face.
    """
    tri, _, inside = locate(mesh, q)
    g = face_gradient(mesh, d)[tri]
    n = jnp.concatenate([g, jnp.ones_like(g[..., :1])], axis=-1)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    return jnp.where(inside[..., None], n, jnp.nan)
