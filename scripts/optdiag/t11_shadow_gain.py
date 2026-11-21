r"""What pulls the gain: the exact gradient in psi with dark blocks scored against the labels' gain or not.

Usage: run.sh t11_shadow_gain.py K STATE

The gradient is exact here (log-kappa modes from a grid, shadow factor differentiated through log-kappa;
`t4_fd.py` checks it against differences). `Problem.shadow_gain` multiplies the predicted block mean by
:math:`g / g_\mathrm{label}`, so a gain raised after the labels were taken predicts brighter dark blocks.
"""
import dataclasses
import sys

import numpy as np
from common import log, setup

from bbf.inference.objective import data_term, value_and_grad

k, state = int(sys.argv[1]), sys.argv[2]
s = setup(k, state)
names = ["port centre", "port width", "port g1", "stbd centre", "stbd width", "stbd g1"]
for flag in (False, True):
    pb = dataclasses.replace(s.problem, mode_grid=49, shadow_stop=False, shadow_gain=flag)
    ell, _, det = value_and_grad(pb, s.d, s.beta, data_term(pb), psi=s.psi, xi=s.xi)
    g = det["grad_psi"].ravel()
    log(f"shadow_gain {flag}: value {ell:.1f}; gradient in psi (per rad, rad, dB/m): "
        + ", ".join(f"{nm} {v:+.0f}" for nm, v in zip(names, g)))
    for j in (2, 5):
        e = np.zeros(6)
        e[j] = 0.01
        up = value_and_grad(pb, s.d, s.beta, data_term(pb), psi=s.psi + e.reshape(2, 3), xi=s.xi)[0]
        log(f"    value at {names[j]} + 0.01 dB/m: {up - ell:+.1f} (gradient predicts {g[j] * 0.01:+.1f})")
