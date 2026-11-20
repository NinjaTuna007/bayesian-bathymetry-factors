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
import math
from dataclasses import dataclass
from functools import cached_property

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


@dataclass(frozen=True)
class Prior:
    r"""The two prior factors :math:`p(\beta)` and :math:`p(d \mid \beta)`; mesh, :math:`\rho`, :math:`\sigma`, :math:`\mu_\beta`, :math:`\Sigma_\beta` are fixed.

    The trend is planar when :math:`\mu_\beta` has 3 entries and constant when it has 1;
    alpha is the SPDE order (2 or 3).
    """
    mesh: Mesh
    rho: float
    sigma: float
    mu_beta: np.ndarray
    Sigma_beta: np.ndarray
    alpha: int = 2

    @property
    def n_beta(self) -> int:
        return len(np.atleast_1d(self.mu_beta))

    @cached_property
    def Lambda(self) -> sp.csc_matrix:
        return precision(self.mesh, self.rho, self.sigma, self.alpha)

    @cached_property
    def X(self) -> np.ndarray:
        return trend_design(self.mesh, self.n_beta)

    @cached_property
    def P_beta(self) -> np.ndarray:
        return np.linalg.inv(np.atleast_2d(self.Sigma_beta))

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

    With :math:`\Lambda = \tau^2 K C^{-1} K`, :math:`u = K^{-1} C^{1/2} z / \tau` with
    :math:`z \sim \mathcal{N}(0, I)` has covariance :math:`\Lambda^{-1}`, one sparse solve with
    :math:`K`. For :math:`\alpha = 3` the odd power of :math:`K` has no such square root, so
    :math:`u = P^\top L^{-\top} z` from the sparse Cholesky factor :math:`P \Lambda P^\top = L L^\top`.
    """
    rng = np.random.default_rng(rng)
    if beta is None:
        beta = np.atleast_1d(rng.multivariate_normal(np.atleast_1d(prior.mu_beta), np.atleast_2d(prior.Sigma_beta)))
    tau = math.sqrt(_tau2(prior.rho, prior.sigma, prior.alpha))
    z = rng.standard_normal(prior.mesh.n_vertices)
    if prior.alpha == 3:
        f = cho_factor(prior.Lambda)
        y = f.solve(z, system="Lt")
        u = np.empty_like(y)
        u[np.asarray(f.perm)] = y
    else:
        K = operator(prior.mesh, prior.rho, prior.alpha)
        u = spla.spsolve(K, np.sqrt(mass(prior.mesh).diagonal()) * z) / tau
    return prior.X @ beta + u, beta
