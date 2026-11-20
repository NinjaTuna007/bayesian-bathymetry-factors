r"""Sonar parameters calibrated by the prototype (scripts/match_factor.py THETA_*, patch A, 2026-11-18).

:math:`L`, :math:`N_c`, :math:`\epsilon_c` and the gain; the spike level, ringing and the
prior on :math:`\log\kappa` were not kept and are fitted where needed.
"""
import numpy as np

GAIN_DB = np.array([[0.0, 0.65, 0.0], [0.0, 0.0, 4.1]])
L_SHAPE = 0.48
NOISE = np.array([9.33e8, 5.28e7])
EPS = np.array([0.055, 0.021])
