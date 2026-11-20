r"""Intrinsic parameters :math:`\theta` of the side-scan sonar, fixed during inference.

Side index 0 is port and 1 is starboard throughout; per-side arrays have shape (2,).
"""
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

PORT, STARBOARD = 0, 1


class SideTheta(NamedTuple):
    """The per-side parameters of one side, as floats; `gain_db` is (3,)."""
    noise: float
    eps: float
    eta: float
    zeta: float
    ell: float
    log_kappa_mean: float
    log_kappa_std: float
    gain_db: np.ndarray


def _pair(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return np.broadcast_to(a, (2,)).copy() if a.ndim == 0 else a


@dataclass(frozen=True)
class Theta:
    r"""Parameters of the predicted intensity and the bin mixture.

    Per-side fields take a scalar (same on both sides) or a pair; `at` reads one side.

    Beam pattern :math:`B(\beta) = \exp(-4 \ln 2\, (\beta - \beta_0)^2 / \omega^2)`, absorption
    :math:`\alpha` (amplitude, Np/m, so the two-way intensity loss is :math:`e^{-4 \alpha r}`),
    receiver gain in dB of amplitude
    :math:`10 \log_{10} g_c(r) = g_{c,0} + g_{c,1} r + g_{c,2} \log_{10} r` (`gain_db`, rows
    port and starboard), background noise :math:`N_c` (intensity before gain), gamma shape
    :math:`L`, spike probability :math:`\epsilon_c`, spike level :math:`\eta_c`, ringing
    level :math:`\zeta_c` and decay length :math:`\ell_c`, and the Gaussian prior on
    :math:`\log \kappa_{t,c}` (mean `log_kappa_mean`, standard deviation `log_kappa_std`).

    The defaults for :math:`\beta_0`, :math:`\omega`, :math:`\alpha` are the PingDSP 3DSS
    values of the prototype (50° below horizontal, 60°, pure water at 450 kHz).

    Shadows: visibility softness :math:`\tau` (`shadow_tau`, rad) and self-shadow margin
    (`shadow_margin`, m) of `geometry.horizon_visibility`; lit/shadow labels are taken over
    blocks of `block` bins, and a noise-only block is labelled lit with probability
    `label_fa`.
    """
    noise: npt.ArrayLike
    L: float
    eps: npt.ArrayLike
    eta: npt.ArrayLike
    zeta: npt.ArrayLike
    ell: npt.ArrayLike
    log_kappa_mean: npt.ArrayLike
    log_kappa_std: npt.ArrayLike
    gain_db: npt.ArrayLike = field(default_factory=lambda: np.zeros((2, 3)))
    beta0: float = math.radians(50.0)
    omega: float = math.radians(60.0)
    alpha: float = 0.0092
    shadow_tau: float = 5e-4
    shadow_margin: float = 0.5
    block: int = 8
    label_fa: float = 0.01

    def __post_init__(self):
        for name in ("noise", "eps", "eta", "zeta", "ell", "log_kappa_mean", "log_kappa_std"):
            object.__setattr__(self, name, _pair(getattr(self, name)))
        object.__setattr__(self, "gain_db", np.asarray(self.gain_db, dtype=float).reshape(2, 3))

    def at(self, side: int) -> SideTheta:
        """Per-side parameters of `side` (0 port, 1 starboard)."""
        def f(name):
            return float(np.asarray(getattr(self, name))[side])
        return SideTheta(f("noise"), f("eps"), f("eta"), f("zeta"), f("ell"), f("log_kappa_mean"),
                         f("log_kappa_std"), np.asarray(self.gain_db)[side])
