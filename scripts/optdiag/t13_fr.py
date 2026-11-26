r"""How far does the first-return factor's quadratic model hold? Its own Newton step, scaled, against the exact change.

Usage: run.sh t13_fr.py K STATE [SIGMA]

The step :math:`\delta = (\Lambda + A^\top W A)^{-1} g_\mathrm{fr}` (the prior's precision keeps it smooth),
scaled so its largest vertex move is 0.03 to 1 m; prints the factor's exact gain, the gain its GGN
predicts, by pass and side. SIGMA replaces the narrow component's standard deviation (default 0.15 m).
"""
import dataclasses
import sys

import numpy as np
import scipy.sparse as sp
from common import log, setup
from sksparse.cholmod import cho_factor

from bbf.ping.first_return import first_return_term

k, state = int(sys.argv[1]), sys.argv[2]
s = setup(k, state)
fr = s.first_return if len(sys.argv) < 4 else dataclasses.replace(s.first_return, sigma=float(sys.argv[3]))
pb = s.problem
n = pb.mesh.n_vertices
v0, g, A, W = first_return_term(pb.mesh, s.d, pb.poses, fr)
H = sp.csc_matrix(A.T @ sp.diags(W) @ A)
delta = cho_factor(sp.csc_matrix(pb.prior.Lambda + H)).solve(g)
log(f"first-return factor at k={k} {state}, sigma {fr.sigma}: value {v0:.1f}, {len(W)} rows; Newton step max "
    f"{np.abs(delta).max():.2f} m, predicted gain {g @ delta - 0.5 * delta @ (H @ delta):.1f}")
for size in (0.03, 0.1, 0.3, 1.0):
    z = delta * size / np.abs(delta).max()
    v1 = first_return_term(pb.mesh, s.d + z, pb.poses, fr)[0]
    pred = g @ z - 0.5 * z @ (H @ z)
    log(f"  largest move {size:4.2f} m: actual {v1 - v0:+9.1f}, predicted {pred:+9.1f}, linear term {g @ z:+9.1f}, "
        f"ratio {(v1 - v0) / pred:5.2f}")
