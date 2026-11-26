r"""GMRF prior on the vertex depths: a trend plus a Matérn field (SPDE, Lindgren et al. 2011).

.. math::

    \beta \sim \mathcal{N}(\mu_\beta, \Sigma_\beta), \qquad
    d \mid \beta \sim \mathcal{N}(X\beta, \Lambda^{-1}),

with rows :math:`(x, y, 1)` of :math:`X` in mesh-local coordinates for a planar trend, or
:math:`(1)` for a constant mean (set by the length of :math:`\mu_\beta`), :math:`C` the lumped
mass matrix and :math:`S` the stiffness matrix of the hat functions. With
:math:`\kappa^2 = 8\nu / \rho^2` and :math:`K = \kappa^2 C + S`, the SPDE of order :math:`\alpha`
(smoothness :math:`\nu = \alpha - 1` in 2D) gives

.. math::

    \alpha = 2: \; \Lambda = \tau^2 K C^{-1} K, \qquad
    \alpha = 3: \; \Lambda = \tau^2 K C^{-1} K C^{-1} K, \qquad
    \tau^2 = \frac{\Gamma(\nu)}{\Gamma(\alpha) (4\pi) \kappa^{2\nu} \sigma^2},

:math:`\rho` the range (correlation :math:`\approx 0.14` at distance :math:`\rho`) and
:math:`\sigma` the marginal standard deviation away from the boundary. Lumping :math:`C` keeps
:math:`\Lambda` sparse. With :math:`\alpha = 2` (:math:`\nu = 1`) the field is not mean-square
differentiable: the prior variance of its slope grows without bound as the mesh is refined,
so the prior on the face normals the sonar sees depends on :math:`h`. :math:`\alpha = 3`
(:math:`\nu = 2`) has well-defined slopes.

On the equilateral lattice every triangle has area :math:`\frac{\sqrt{3}}{4} h^2` and
angles of :math:`60^\circ`, so

.. math::

    C_{ii} = k_i \frac{\sqrt{3}}{12} h^2, \qquad
    S_{ij} = -\frac{m_{ij}}{2\sqrt{3}}, \qquad
    S_{ii} = -\sum_{j \ne i} S_{ij},

with :math:`k_i` the number of triangles at vertex :math:`i` and :math:`m_{ij}` the number
of triangles on edge :math:`ij`.
"""
import dataclasses
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Self

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import cho_factor

from bbf.mesh import Mesh


def mass(mesh: Mesh) -> sp.csc_matrix:
    r"""Lumped (diagonal) mass matrix :math:`C`."""
    k = np.bincount(mesh.triangles.ravel(), minlength=mesh.n_vertices)
    return sp.csc_matrix(sp.diags(k * (math.sqrt(3) / 12) * mesh.h**2))


def stiffness(mesh: Mesh) -> sp.csc_matrix:
    r"""Stiffness matrix :math:`S_{ij} = \int_\Omega \nabla\psi_i \cdot \nabla\psi_j \, dq`."""
    t = mesh.triangles
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    w = np.full(i.size, -1 / (2 * math.sqrt(3)))
    off = sp.coo_matrix((w, (i, j)), shape=(mesh.n_vertices,) * 2)
    off = (off + off.T).tocsc()  # duplicates sum: shared edges get :math:`m_{ij} = 2`
    return (off - sp.diags(np.asarray(off.sum(axis=1)).ravel())).tocsc()


def operator(mesh: Mesh, rho: float, alpha: int = 2) -> sp.csc_matrix:
    r""":math:`K = \kappa^2 C + S` with :math:`\kappa^2 = 8\nu / \rho^2`, :math:`\nu = \alpha - 1`."""
    return ((8 * (alpha - 1) / rho**2) * mass(mesh) + stiffness(mesh)).tocsc()


def _tau2(rho: float, sigma: float, alpha: int) -> float:
    nu = alpha - 1
    kappa2 = 8 * nu / rho**2
    return math.gamma(nu) / (math.gamma(alpha) * 4 * math.pi * kappa2**nu * sigma**2)


def precision(mesh: Mesh, rho: float, sigma: float, alpha: int = 2) -> sp.csc_matrix:
    r"""Sparse precision :math:`\Lambda(\rho, \sigma)` of the Matérn part, SPDE order :math:`\alpha \in \{2, 3\}`."""
    if alpha not in (2, 3):
        raise ValueError("alpha must be 2 or 3")
    K = operator(mesh, rho, alpha)
    Cinv = sp.diags(1 / mass(mesh).diagonal())
    Q = K @ Cinv @ K
    if alpha == 3:
        Q = Q @ Cinv @ K
    return (_tau2(rho, sigma, alpha) * Q).tocsc()


def trend_design(mesh: Mesh, p: int = 3) -> np.ndarray:
    r"""Trend design :math:`X` of shape (n, p): rows :math:`(x, y, 1)` (planar, p = 3) or :math:`(1)` (constant, p = 1)."""
    v = mesh.vertices
    if p == 1:
        return np.ones((len(v), 1))
    if p == 3:
        return np.column_stack([v, np.ones(len(v))])
    raise ValueError("the trend has 1 (constant) or 3 (planar) coefficients")


def bowl_design(mesh: Mesh, tracks: np.ndarray, scale: float = 30.0) -> np.ndarray:
    r"""Trend design (n, 2) of a channel: rows :math:`(1, -\min(s, s_0)^2 / s_0^2)`, :math:`s` the distance to the nearest track point.

    tracks: (m, 2) sonar positions in mesh-local coordinates; scale: :math:`s_0` (m). With
    :math:`\beta = (\beta_0, \beta_1)` the mean depth is :math:`\beta_0` under the tracks and rises by
    :math:`\beta_1` metres to a rim at :math:`s_0` and beyond: a bowl across the track for
    :math:`\beta_1 > 0`, the constant mean for :math:`\beta_1 = 0`. A boat keeps to the deep water, so
    where the seabed shoals to a shore it does so away from the tracks; with the constant mean the
    field relaxes to the depth under the tracks within its range, a flat floor where the banks are.
    """
    from scipy.spatial import cKDTree

    s = cKDTree(np.asarray(tracks, dtype=float)).query(mesh.vertices)[0]
    return np.column_stack([np.ones(mesh.n_vertices), -np.minimum(s, scale) ** 2 / scale**2])


@dataclass(frozen=True)
class Prior:
    r"""The two prior factors :math:`p(\beta)` and :math:`p(d \mid \beta)`; mesh, :math:`\rho`, :math:`\sigma`, :math:`\mu_\beta`, :math:`\Sigma_\beta` are fixed.

    The trend is planar when :math:`\mu_\beta` has 3 entries and constant when it has 1, or has
    the columns of `design` (n, p) when that is given (e.g. `bowl_design`);
    alpha is the SPDE order (2 or 3).

    slope, if given, adds a factor on the facet directions: each face's gradient :math:`\nabla u_f` of
    :math:`u = d - X\beta` is :math:`\mathcal{N}(0, s^2 I)`, so

    .. math::

        \Lambda \leftarrow \Lambda + \frac{S}{A s^2}, \qquad A = \tfrac{\sqrt{3}}{4} h^2,

    with :math:`S` the stiffness matrix (:math:`u^\top S u = A \sum_f \lVert \nabla u_f \rVert^2` on the equilateral
    lattice). The Matérn part with :math:`\alpha = 2` leaves the slopes at the mesh scale free, and the
    echo's :math:`\cos^2\varphi` then tilts single faces to explain brightness.

    slope_nu, if given with slope, makes that factor heavy-tailed: each face's gradient is a
    bivariate Student-t with :math:`\nu` degrees of freedom,

    .. math::

        \log p(\nabla u_f) = -\frac{\nu + 2}{2} \log\Bigl(1 + \frac{\lVert \nabla u_f \rVert^2}{(\nu + 2) s^2}\Bigr),

    scaled so that its curvature at a level face is the Gaussian's, :math:`1 / s^2`. Mesh-scale
    roughness costs about what it did, while a bank costs only the logarithm of its slope: at
    :math:`s = 0.2`, :math:`\nu = 1` a face of :math:`40^\circ` costs 2.9 instead of 8.8. The factor is not quadratic,
    so `Lambda` is its iteratively reweighted curvature at the gradients of `slope_at` (`at`),

    .. math::

        \Lambda = \Lambda_\text{Matérn} + \sum_f w_f\, G_f^\top G_f, \qquad w_f = \frac{\nu + 2}{(\nu + 2) s^2 + \lVert \nabla u_f \rVert^2},

    with :math:`G_f` the face's gradient operator; at a level `slope_at` (the default) it is the
    Gaussian factor's. `log_density` and `grad` are exact; with `slope_at` at the same point,
    :math:`-\Lambda u` is the gradient in :math:`u`.
    """
    mesh: Mesh
    rho: float
    sigma: float
    mu_beta: np.ndarray
    Sigma_beta: np.ndarray
    alpha: int = 2
    slope: float | None = None
    slope_nu: float | None = None
    slope_at: np.ndarray | None = field(default=None, compare=False, repr=False)
    design: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def n_beta(self) -> int:
        return len(np.atleast_1d(self.mu_beta))

    @cached_property
    def Lambda(self) -> sp.csc_matrix:
        Q = self.Lambda_matern
        if self.slope is not None and self.slope_nu is None:
            Q = Q + stiffness(self.mesh) / (math.sqrt(3) / 4 * self.mesh.h**2 * self.slope**2)
        elif self.slope is not None:
            gx, gy = self._face_gradient(np.zeros(self.mesh.n_vertices) if self.slope_at is None else self.slope_at)
            w = self._slope_weights(gx, gy)
            Gx, Gy = self._grad_rows
            Q = Q + Gx.T @ sp.diags(w) @ Gx + Gy.T @ sp.diags(w) @ Gy
        return sp.csc_matrix(Q)

    @cached_property
    def Lambda_matern(self) -> sp.csc_matrix:
        r"""The Matérn part of the precision alone."""
        return precision(self.mesh, self.rho, self.sigma, self.alpha).tocsc()

    @cached_property
    def _grad_rows(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        r"""Sparse (m, n) maps from vertex values to the x and y components of each face's gradient."""
        t = np.asarray(self.mesh.triangles)
        ops = np.asarray(self.mesh._grad_ops)  # (m, 2, 2) on (d_1 - d_0, d_2 - d_0)
        m = len(t)
        rows = np.repeat(np.arange(m), 3)
        out = []
        for i in (0, 1):
            vals = np.stack([-ops[:, i, 0] - ops[:, i, 1], ops[:, i, 0], ops[:, i, 1]], axis=1).ravel()
            out.append(sp.csr_matrix((vals, (rows, t.ravel())), shape=(m, self.mesh.n_vertices)))
        return out[0], out[1]

    def _face_gradient(self, u):
        Gx, Gy = self._grad_rows
        return Gx @ u, Gy @ u

    def _slope_weights(self, gx, gy):
        nu, s = self.slope_nu, self.slope
        return (nu + 2) / ((nu + 2) * s**2 + gx**2 + gy**2)

    def at(self, d: np.ndarray, beta: np.ndarray) -> Self:
        r"""This prior with the heavy-tailed slope factor's curvature taken at (d, :math:`\beta`); itself otherwise."""
        if self.slope is None or self.slope_nu is None:
            return self
        return dataclasses.replace(self, slope_at=np.asarray(d, float) - self.X @ np.asarray(beta, float))

    @cached_property
    def X(self) -> np.ndarray:
        if self.design is not None:
            if self.design.shape != (self.mesh.n_vertices, self.n_beta):
                raise ValueError(f"design is {self.design.shape}, expected {(self.mesh.n_vertices, self.n_beta)}")
            return np.asarray(self.design, dtype=float)
        return trend_design(self.mesh, self.n_beta)

    @cached_property
    def P_beta(self) -> np.ndarray:
        return np.linalg.inv(np.atleast_2d(self.Sigma_beta))

    def log_density(self, d: np.ndarray, beta: np.ndarray) -> float:
        r""":math:`\log p(\beta) + \log p(d \mid \beta)`, up to a constant independent of :math:`d` and :math:`\beta`."""
        u, b = d - self.X @ beta, beta - self.mu_beta
        if self.slope is not None and self.slope_nu is not None:
            gx, gy = self._face_gradient(u)
            nu = self.slope_nu
            lt = -0.5 * (nu + 2) * np.sum(np.log1p((gx**2 + gy**2) / ((nu + 2) * self.slope**2)))
            return float(lt) - 0.5 * float(u @ (self.Lambda_matern @ u) + b @ self.P_beta @ b)
        return -0.5 * float(u @ (self.Lambda @ u) + b @ self.P_beta @ b)

    def grad(self, d: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Gradient of :meth:`log_density` with respect to :math:`(d, \beta)`."""
        u = d - self.X @ beta
        if self.slope is not None and self.slope_nu is not None:
            gx, gy = self._face_gradient(u)
            w = self._slope_weights(gx, gy)
            Gx, Gy = self._grad_rows
            Lu = self.Lambda_matern @ u + Gx.T @ (w * gx) + Gy.T @ (w * gy)
        else:
            Lu = self.Lambda @ u
        return -Lu, self.X.T @ Lu - self.P_beta @ (beta - self.mu_beta)

    def joint_precision(self) -> sp.csc_matrix:
        r"""Precision of :math:`(d, \beta)`, with :math:`\beta` last:

        .. math::

            \begin{pmatrix} \Lambda & -\Lambda X \\
            -X^\top \Lambda & \Sigma_\beta^{-1} + X^\top \Lambda X \end{pmatrix}.


        The prior is quadratic, so this is also its exact (negative) Hessian; with the heavy-tailed
        slope factor it is the reweighted curvature at `slope_at`.
        """
        LX = sp.csc_matrix(self.Lambda @ self.X)
        corner = sp.csc_matrix(self.P_beta + self.X.T @ (self.Lambda @ self.X))
        return sp.csc_matrix(sp.bmat([[self.Lambda, -LX], [-LX.T, corner]]))


def sample(prior: Prior, rng: np.random.Generator | int | None = None, beta: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    r"""Draw :math:`(d, \beta)` from the prior; :math:`\beta` is drawn unless given.

    With :math:`\Lambda = \tau^2 K C^{-1} K`, :math:`u = K^{-1} C^{1/2} z / \tau` with
    :math:`z \sim \mathcal{N}(0, I)` has covariance :math:`\Lambda^{-1}`, one sparse solve with
    :math:`K`. For :math:`\alpha = 3` (and with a slope factor) the odd power of :math:`K` has no such square root, so
    :math:`u = P^\top L^{-\top} z` from the sparse Cholesky factor :math:`P \Lambda P^\top = L L^\top`.
    """
    rng = np.random.default_rng(rng)
    if beta is None:
        beta = np.atleast_1d(rng.multivariate_normal(np.atleast_1d(prior.mu_beta), np.atleast_2d(prior.Sigma_beta)))
    tau = math.sqrt(_tau2(prior.rho, prior.sigma, prior.alpha))
    z = rng.standard_normal(prior.mesh.n_vertices)
    if prior.alpha == 3 or prior.slope is not None:
        f = cho_factor(prior.Lambda)
        y = f.solve(z, system="Lt")
        u = np.empty_like(y)
        u[np.asarray(f.perm)] = y
    else:
        K = operator(prior.mesh, prior.rho, prior.alpha)
        u = spla.spsolve(K, np.sqrt(mass(prior.mesh).diagonal()) * z) / tau
    return prior.X @ beta + u, beta
