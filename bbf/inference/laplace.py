r"""Laplace approximation: posterior variances of the depths from the Gauss--Newton precision.

At the mode, the posterior of all unknowns is approximated by :math:`\mathcal{N}(\hat z, H^{-1})`
with :math:`H` the augmented GGN precision (`bbf.inference.ggn`). Because :math:`u = \log\kappa` and
the log-reflectivity field :math:`\xi` are rows of :math:`H`, the marginal of :math:`d` in :math:`H^{-1}`
already integrates them out. The variances :math:`[H^{-1}]_{ii}` are estimated without forming :math:`H^{-1}`, by
Rao--Blackwellised Monte Carlo (Sidén, Lindgren, Bolin, Villani 2018): with samples
:math:`x \sim \mathcal{N}(0, H^{-1})` from the Cholesky factor,

.. math::

    [H^{-1}]_{ii} = \frac{1}{H_{ii}} + \mathrm{Var}\bigl(\mathbb{E}[x_i \mid x_{-i}]\bigr)
    \approx \frac{1}{H_{ii}} + \frac{1}{S} \sum_{s} \Bigl(\frac{1}{H_{ii}} \sum_{j \ne i} H_{ij} x^{(s)}_j\Bigr)^2,

whose error is far smaller than that of the plain sample variance for the same samples.
"""
import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cho_factor

from bbf.inference import ggn
from bbf.inference.objective import Problem, value_and_grad


def sample_precision(H, n_samples: int, rng=None) -> np.ndarray:
    r"""Samples (size, n_samples) of :math:`\mathcal{N}(0, H^{-1})` for a sparse SPD precision H."""
    f = cho_factor(sp.csc_matrix(H))
    w = np.random.default_rng(rng).standard_normal((H.shape[0], n_samples))
    y = f.solve(w, system="Lt")  # :math:`L^\top y = w` in the factor's permuted order: :math:`y \sim \mathcal{N}(0, (P H P^\top)^{-1})`
    x = np.empty_like(y)
    x[np.asarray(f.perm)] = y
    return x


def marginal_variances(H, n_samples: int = 200, rng=None) -> np.ndarray:
    r""":math:`[H^{-1}]_{ii}` for every i, by Rao--Blackwellised Monte Carlo with `n_samples` samples."""
    H = sp.csr_matrix(H)
    X = sample_precision(H, n_samples, rng)
    Hd = H.diagonal()
    m = (H @ X - Hd[:, None] * X) / Hd[:, None]  # E[x_i | x_-i] up to sign, per sample
    return 1.0 / Hd + np.mean(m**2, axis=1)


def laplace(problem: Problem, d, beta, J, n_samples: int = 200, rng=None, Jb=None, psi=None, xi=None) -> dict:
    r"""Posterior standard deviations at the mode (d, :math:`\beta`, :math:`\psi`, :math:`\xi`) with frozen Jacobians J (and Jb).

    Returns the depth standard deviations (n,), those of :math:`\beta` (3, or 1),, of :math:`\psi` (2, 3)
    when it is free (zero for its held components), of :math:`\xi` (n,) when the problem has the field, and the augmented
    precision H it came from (undamped).
    """
    _, _, det = value_and_grad(problem, d, beta, psi=psi, xi=xi)
    H = ggn.assemble(problem, J, det["S"], det["u"], Jb=Jb, psi=psi, fr=det["first_return"])
    lay = ggn.layout(problem)
    free = np.ones(lay["size"], bool)
    if problem.free_psi and problem.psi_held is not None:
        free[lay["psi"]:lay["psi"] + 6] = ~np.asarray(problem.psi_held, bool).ravel()
    var = np.zeros(lay["size"])  # held sonar parameters are known: no variance, and conditioned on
    var[free] = marginal_variances(sp.csr_matrix(H)[free][:, free], n_samples, rng)
    n, ib, ip = lay["n"], lay["beta"], lay["psi"]
    out = {"sd_d": np.sqrt(var[:n]), "sd_beta": np.sqrt(var[ib:ib + lay["n_beta"]]), "H": H}
    if problem.refl is not None:
        out["sd_xi"] = np.sqrt(var[lay["xi"]:lay["xi"] + n])
    if problem.free_psi:
        out["sd_psi"] = np.sqrt(var[ip:ip + 6]).reshape(2, 3)
    return out
