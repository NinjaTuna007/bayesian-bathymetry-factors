r"""Sonar parameters: the 3DSS's own gain law with its standard settings, and the water column's background.

The 3DSS applies :math:`G(r) = \mathrm{LOG} \log_{10} r + \mathrm{LINEAR}\, r + \mathrm{CONSTANT}` dB
(UM001 3DSS-DX/iDX Sonar Manual 2.3, "Gain parameters"); its standard settings, which the normal GUI sets
and the manual and command guide use throughout, are LOG = 30, LINEAR = 0.2, CONSTANT = 0 on both sides.
The survey's own values are not in our bags, so the standard ones are used as they are, never fitted.
In natural log of intensity this is :math:`3 \log r + c\, r` with :math:`c = 0.2 \ln 10 / 10`; the
constant is normalised to one at 20 m, which `Theta.noise` and the log-:math:`\kappa` prior absorb.

The background is volume reverberation measured on starboard's water column after gain
(`scripts/residual_background.py`, passes A and B of the 2026-07-10 bag), :math:`W \propto (r / 20)^{w} P`
with :math:`w = 1.48`; before gain it is :math:`N (r / 20)^{\nu} P` with
:math:`\nu = w - 3 - 20 c` to first order about 20 m, and :math:`N = W_{20}` at full power. Port's water
column agrees beyond its sensor-fixed lines (within 3 dB beyond 10 m on B), which `MIN_RANGE` leaves out.
"""
import math

import numpy as np

LOG_DB, LINEAR_DB = 30.0, 0.2  # the 3DSS's standard gain (dB per decade, dB/m), both sides
_C = LINEAR_DB * math.log(10) / 10  # :math:`c`, the linear gain in natural log of intensity per metre
GAIN_DB = np.array([[-(LINEAR_DB * 20.0 + LOG_DB * math.log10(20.0)), LINEAR_DB, LOG_DB]] * 2)
NOISE = np.array([8.53e8, 8.53e8])
WATER_SLOPE = 1.48  # :math:`w`, measured after gain
NOISE_SLOPE = WATER_SLOPE - LOG_DB / 10 - 20 * _C  # :math:`\nu`
NOISE_POWER = 1.0
MIN_RANGE = 8.0  # port's lines reach 7.8 m on B
LOG_KAPPA_MEAN = 39.0
