r"""GMRF prior on the vertex depths: planar trend plus a Matérn field (SPDE, Lindgren et al. 2011).

.. math::

    \beta \sim \mathcal{N}(\mu_\beta, \Sigma_\beta), \qquad
    d \mid \beta \sim \mathcal{N}(X\beta, \Lambda^{-1}), \\
    \Lambda = \frac{\rho^2}{32\pi\sigma^2} K C^{-1} K, \qquad
    K = \frac{8}{\rho^2} C + S,

with rows :math:`(x, y, 1)` of :math:`X` in mesh-local coordinates, :math:`C` the lumped
mass matrix and :math:`S` the stiffness matrix of the hat functions. This is
:math:`\alpha = 2` (smoothness :math:`\nu = 1`) in 2D: :math:`\rho` is the range
(correlation :math:`\approx 0.14` at distance :math:`\rho`) and :math:`\sigma` the marginal
standard deviation away from the boundary. Lumping :math:`C` keeps :math:`\Lambda` sparse.

On the equilateral lattice every triangle has area :math:`\frac{\sqrt{3}}{4} h^2` and
angles of :math:`60^\circ`, so

.. math::

    C_{ii} = k_i \frac{\sqrt{3}}{12} h^2, \qquad
    S_{ij} = -\frac{m_{ij}}{2\sqrt{3}}, \qquad
    S_{ii} = -\sum_{j \ne i} S_{ij},

with :math:`k_i` the number of triangles at vertex :math:`i` and :math:`m_{ij}` the number
of triangles on edge :math:`ij`.
"""
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


def operator(mesh: Mesh, rho: float) -> sp.csc_matrix:
    r""":math:`K = \frac{8}{\rho^2} C + S`."""
    return ((8 / rho**2) * mass(mesh) + stiffness(mesh)).tocsc()


def precision(mesh: Mesh, rho: float, sigma: float) -> sp.csc_matrix:
    r"""Sparse precision :math:`\Lambda(\rho, \sigma)` of the Matérn part."""
    K = operator(mesh, rho)
    Cinv = sp.diags(1 / mass(mesh).diagonal())
    return (rho**2 / (32 * math.pi * sigma**2) * (K @ Cinv @ K)).tocsc()


def trend_design(mesh: Mesh) -> np.ndarray:
    r"""Trend design :math:`X` of shape (n, 3), rows :math:`(x, y, 1)` per vertex in local coordinates."""
    v = mesh.vertices
    return np.column_stack([v, np.ones(len(v))])


@dataclass(frozen=True)
class Prior:
    r"""The two prior factors :math:`p(\beta)` and :math:`p(d \mid \beta)`; mesh, :math:`\rho`, :math:`\sigma`, :math:`\mu_\beta`, :math:`\Sigma_\beta` are fixed."""
    mesh: Mesh
    rho: float
    sigma: float
    mu_beta: np.ndarray
    Sigma_beta: np.ndarray

    @cached_property
    def Lambda(self) -> sp.csc_matrix:
        return precision(self.mesh, self.rho, self.sigma)

    @cached_property
    def X(self) -> np.ndarray:
        return trend_design(self.mesh)

    @cached_property
    def P_beta(self) -> np.ndarray:
        return np.linalg.inv(self.Sigma_beta)

    def log_density(self, d: np.ndarray, beta: np.ndarray) -> float:
        r""":math:`\log p(\beta) + \log p(d \mid \beta)`, up to a constant independent of :math:`d` and :math:`\beta`."""
        u, b = d - self.X @ beta, beta - self.mu_beta
        return -0.5 * float(u @ (self.Lambda @ u) + b @ self.P_beta @ b)

    def grad(self, d: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Gradient of :meth:`log_density` with respect to :math:`(d, \beta)`."""
        Lu = self.Lambda @ (d - self.X @ beta)
        return -Lu, self.X.T @ Lu - self.P_beta @ (beta - self.mu_beta)

    def joint_precision(self) -> sp.csc_matrix:
        r"""Precision of :math:`(d, \beta)`, with :math:`\beta` last:

        .. math::

            \begin{pmatrix} \Lambda & -\Lambda X \\
            -X^\top \Lambda & \Sigma_\beta^{-1} + X^\top \Lambda X \end{pmatrix}.


        The prior is quadratic, so this is also its exact (negative) Hessian.
        """
        LX = sp.csc_matrix(self.Lambda @ self.X)
        corner = sp.csc_matrix(self.P_beta + self.X.T @ (self.Lambda @ self.X))
        return sp.csc_matrix(sp.bmat([[self.Lambda, -LX], [-LX.T, corner]]))


def sample(prior: Prior, rng: np.random.Generator | int | None = None, beta: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    r"""Draw :math:`(d, \beta)` from the prior; :math:`\beta` is drawn unless given.

    Uses :math:`\Lambda = \tau^2 K C^{-1} K`: :math:`u = K^{-1} C^{1/2} z / \tau` with
    :math:`z \sim \mathcal{N}(0, I)` has covariance :math:`\Lambda^{-1}`, one sparse solve
    with :math:`K`.
    """
    rng = np.random.default_rng(rng)
    if beta is None:
        beta = rng.multivariate_normal(prior.mu_beta, prior.Sigma_beta)
    tau = prior.rho / math.sqrt(32 * math.pi) / prior.sigma
    z = rng.standard_normal(prior.mesh.n_vertices)
    u = spla.spsolve(operator(prior.mesh, prior.rho), np.sqrt(mass(prior.mesh).diagonal()) * z) / tau
    return prior.X @ beta + u, beta
