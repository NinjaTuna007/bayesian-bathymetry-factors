r"""Intrinsic parameters :math:`\theta` of the side-scan sonar.

Side index 0 is port and 1 is starboard throughout; per-side arrays have shape (2,).

The parameters that shape the brightness across the swath, the beam centre :math:`\beta_0`,
the beam width :math:`\omega` and the linear range gain :math:`g_1` of each side, form
:math:`\psi` (`psi_of`, `with_psi`), which inference can treat as unknowns in the graph; the
rest is calibrated between rounds or held fixed.
"""
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

PORT, STARBOARD = 0, 1
BEAM_CENTRE, BEAM_WIDTH = math.radians(50.0), math.radians(60.0)  # PingDSP 3DSS defaults (rad)
NOISE_REF = 20.0  # range :math:`r_N` (m) at which the background is `Theta.noise` (see `Theta.noise_slope`)


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


def _pair(v):
    if isinstance(v, jax.Array):  # traced inside a compiled function: keep it traced
        return jnp.broadcast_to(v, (2,))
    a = np.array(v, dtype=float)  # a copy: theta is compiled into functions as constants
    return np.broadcast_to(a, (2,)).copy() if a.ndim == 0 else a


@dataclass(frozen=True, eq=False)  # hashed by identity: its array fields have no hash, and compiled code is cached per theta
class Theta:
    r"""Parameters of the predicted intensity and the bin mixture.

    Per-side fields take a scalar (same on both sides) or a pair; `at` reads one side.

    Beam pattern of side :math:`c` :math:`B_c(\beta) = \exp(-4 \ln 2\, (\beta - \beta_{0,c})^2 / \omega_c^2)`, absorption
    :math:`\alpha` (amplitude, Np/m, so the two-way intensity loss is :math:`e^{-4 \alpha r}`),
    receiver gain in dB of amplitude
    :math:`10 \log_{10} g_c(r) = g_{c,0} + g_{c,1} r + g_{c,2} \log_{10} r` (`gain_db`, rows
    port and starboard), background :math:`N_c` (intensity before gain), gamma shape
    :math:`L`, spike probability :math:`\epsilon_c`, spike level :math:`\eta_c`, ringing
    level :math:`\zeta_c` and decay length :math:`\ell_c`, and the Gaussian prior on
    :math:`\log \kappa_{t,c}` (mean `log_kappa_mean`, standard deviation `log_kappa_std`).

    The defaults for :math:`\beta_0`, :math:`\omega`, :math:`\alpha` are the PingDSP 3DSS
    values of the prototype (:math:`50^\circ` below horizontal, :math:`60^\circ`, pure water at 450 kHz).

    Shadows: acoustic wavelength :math:`\lambda` (`wavelength`, m; 1480 m/s at 450 kHz) for
    the knife-edge diffraction and the self-shadow margin (`shadow_margin`, m) of
    `geometry.horizon_visibility`; lit/shadow labels are taken over
    blocks of `block` bins, and a block of pure noise is labelled lit with probability
    `label_fa`. `wc_lit` (2, n_blocks), if given, is the measured lit rate of blocks with no
    seabed echo by range (ringing, volume scatter, spikes), from
    `bbf.ping.shadow.water_column_lit_rate`; otherwise that rate is
    :math:`\epsilon_B + (1 - \epsilon_B) p_\mathrm{fa}` with :math:`\epsilon_B = 1 - (1 - \epsilon_c)^B`.

    `z_water` is the world height of the water surface: seabed above it returns no echo.

    Background law (`bbf.ping.factor.background`), off by default: the background of a bin at
    range :math:`r` of a ping with log transmit power :math:`o` is
    :math:`N_c (r / r_N)^{\nu} e^{\gamma o}` with :math:`\nu` = `noise_slope`, :math:`\gamma` =
    `noise_power` and :math:`r_N` = `NOISE_REF`. Receiver noise has :math:`\nu = \gamma = 0`;
    volume reverberation follows the transmit power (:math:`\gamma = 1`) and, before a gain
    that cancels the seabed's :math:`r^{-3}`, falls about as :math:`r^{-2}`.
    """
    noise: npt.ArrayLike
    L: float
    eps: npt.ArrayLike
    eta: npt.ArrayLike
    zeta: npt.ArrayLike
    ell: npt.ArrayLike
    log_kappa_mean: npt.ArrayLike
    log_kappa_std: npt.ArrayLike
    gain_db: Any = field(default_factory=lambda: np.zeros((2, 3)))  # (2, 3); may be traced, as beta0
    beta0: Any = BEAM_CENTRE  # per side; Any, since it may be traced inside compiled functions
    omega: Any = BEAM_WIDTH
    alpha: float = 0.0092
    wavelength: float = 1480.0 / 450e3
    shadow_margin: float = 0.5
    block: int = 8
    label_fa: float = 0.01
    wc_lit: npt.ArrayLike | None = None
    z_water: float = math.inf
    noise_slope: float = 0.0
    noise_power: float = 0.0

    def __post_init__(self):
        for name in ("noise", "eps", "eta", "zeta", "ell", "log_kappa_mean", "log_kappa_std", "beta0", "omega"):
            object.__setattr__(self, name, _pair(getattr(self, name)))
        if not isinstance(self.gain_db, jax.Array):
            object.__setattr__(self, "gain_db", np.array(self.gain_db, dtype=float).reshape(2, 3))
        if self.wc_lit is not None:
            object.__setattr__(self, "wc_lit", np.array(self.wc_lit, dtype=float).reshape(2, -1))

    def at(self, side: int) -> SideTheta:
        """Per-side parameters of `side` (0 port, 1 starboard)."""
        def f(name):
            return float(np.asarray(getattr(self, name))[side])
        return SideTheta(f("noise"), f("eps"), f("eta"), f("zeta"), f("ell"), f("log_kappa_mean"),
                         f("log_kappa_std"), self.gain_db[side])


def psi_of(theta: Theta) -> np.ndarray:
    r""":math:`\psi` (2, 3): per side (rows) the beam centre :math:`\beta_0`, width :math:`\omega` (radians) and gain :math:`g_1` (dB/m)."""
    return np.stack([np.asarray(theta.beta0, float), np.asarray(theta.omega, float),
                     np.asarray(theta.gain_db)[:, 1]], axis=1)


def with_psi(theta: Theta, psi) -> Theta:
    r"""theta with :math:`\psi` (2, 3) put in; psi may be traced, so this works inside compiled functions."""
    if isinstance(psi, jax.Array):
        gain = jnp.asarray(theta.gain_db).at[:, 1].set(psi[:, 2])
    else:
        psi = np.asarray(psi, float)
        gain = np.asarray(theta.gain_db, float).copy()
        gain[:, 1] = psi[:, 2]
    return dataclasses.replace(theta, beta0=psi[:, 0], omega=psi[:, 1], gain_db=gain)


def render_key(theta: Theta) -> tuple:
    r"""The fields of theta the rendering of :math:`S` depends on besides :math:`\psi`, as a hashable key."""
    return (float(theta.alpha), float(theta.wavelength), float(theta.shadow_margin), float(theta.z_water))


def render_theta(key: tuple, psi) -> Theta:
    r"""A theta that renders like any theta with this `render_key`, at the sonar parameters :math:`\psi` (may be traced).

    Its noise-model fields are placeholders: only :math:`S` may be computed from it.
    """
    alpha, wavelength, margin, z_water = key
    th = Theta(noise=1.0, L=1.0, eps=0.5, eta=1.0, zeta=1.0, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0,
               alpha=alpha, wavelength=wavelength, shadow_margin=margin, z_water=z_water)
    return with_psi(th, psi)
