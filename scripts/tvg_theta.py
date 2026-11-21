r"""Sonar parameters with one gain law for both sides and the background of the water column.

The gain cancels spreading and absorption of a seabed echo, :math:`g(r) = (r / 20)^3 e^{4 \alpha (r - 20)}`
for both sides (one at 20 m), and the background is volume reverberation measured on starboard's
water column (`scripts/residual_background.py`, passes A and B of the 2026-07-10 bag):
:math:`N (r / 20)^{\nu} P` with :math:`P` the transmit power as a fraction. Port's water column agrees
beyond its sensor-fixed lines (within 3 dB beyond 10 m on B), which `MIN_RANGE` leaves out.
"""
import math

import numpy as np

ALPHA = 0.0092  # absorption (amplitude, Np/m), as `bbf.ping.Theta`'s default
_G1 = 4 * ALPHA * 10 / math.log(10)  # dB/m
GAIN_DB = np.array([[-(_G1 * 20.0 + 30.0 * math.log10(20.0)), _G1, 30.0]] * 2)
NOISE = np.array([8.53e8, 8.53e8])
NOISE_SLOPE = -2.26
NOISE_POWER = 1.0
MIN_RANGE = 8.0  # port's lines reach 7.8 m on B
LOG_KAPPA_MEAN = 39.0
