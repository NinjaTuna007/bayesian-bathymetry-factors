r"""The match factor: one log-reflectivity field on the mesh, shared by every pass.

The seabed at :math:`q` has reflectivity :math:`\kappa(q) = e^{\xi(q)}`, with
:math:`\xi(q) = \sum_i \xi_i \psi_i(q)` on the mesh of the depths and a Matérn prior
(`bbf.prior.spde.precision`),

.. math::

    \xi \sim \mathcal{N}\bigl(0, \Lambda(\rho_\xi, \sigma_\xi)^{-1}\bigr).

The field multiplies the echo inside the render (`bbf.ping.render`), so bin :math:`b` of ping
:math:`t`, side :math:`c` has the mean

.. math::

    \bar I_b = g_c(r_b) \Bigl(\frac{\kappa_{t,c}}{\Delta r}
        \int_{\mathcal{C}_b} e^{\xi(q)} a_c(q)\, V(q)\, \mathrm{d}q + N_c\Bigr),

and :math:`\kappa_{t,c}` is what is left for the ping-side: its gain. Two passes that reach the
same ground see the same :math:`e^{\xi(q)}`, so the ratio of their echoes is free of it and depends
on the depths through the two incidence angles, ranges and beam angles. A slope changes that
ratio; reflectivity cannot. Shading seen by one pass alone can be either, and the two priors
share it out.

This replaces matched pairs of bins. A pair with a common reflectivity under
:math:`p(\kappa_q) \propto 1/\kappa_q` is the marginal of this model for two bins taken to cover
the same ground, with that registration frozen at the depths the pair was chosen at. Here the
footprint of every bin is the render's at the current depths, every bin of every pass takes
part, any number of passes share the field, and no bin is taken out of its ping factor.

Joint estimation of a reflectivity map, the beam pattern and the elevation from side-scan
intensities under Lambert's law goes back to Coiras, Petillot and Lane (2007); here the map is
a variable of the factor graph with a prior, estimated with the depths in one sparse system.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from bbf.mesh import Mesh
from bbf.prior.spde import precision


@dataclass(frozen=True)
class Reflectivity:
    r"""Prior factor :math:`p(\xi)` of the log-reflectivity field: range :math:`\rho_\xi` (m), standard deviation :math:`\sigma_\xi`."""
    mesh: Mesh
    rho: float
    sigma: float

    @cached_property
    def Lambda(self) -> sp.csc_matrix:
        return precision(self.mesh, self.rho, self.sigma)

    def log_density(self, xi: np.ndarray) -> float:
        r""":math:`\log p(\xi)`, up to a constant independent of :math:`\xi`."""
        return -0.5 * float(xi @ (self.Lambda @ xi))

    def grad(self, xi: np.ndarray) -> np.ndarray:
        r"""Gradient of :meth:`log_density`."""
        return -(self.Lambda @ xi)
