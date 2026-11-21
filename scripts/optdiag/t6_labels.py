r"""The data the ping factor scores depend on the gain: lit fraction of the bins under each saved state's psi.

Usage: run.sh t6_labels.py [K]

The lit/shadow labels threshold :math:`y / (g N)`, and `infer` recomputes them every round with the
current gain. Prints, per side and range band, the share of bins labelled lit under the gain of the
start, and of the states after each level of run fr2.
"""
import sys

import numpy as np
from common import CACHE, EPS, GAIN_DB, L_SHAPE, NOISE, RUNS, log

from bbf.inference.loop import labels_of
from bbf.inference.multiscale import coarsen_pass, coarsen_theta
from bbf.io import load_depths, load_patch
from bbf.ping import Theta, bin_ranges, psi_of, with_psi

k = int(sys.argv[1]) if len(sys.argv) > 1 else 27
passes = [coarsen_pass(load_patch(CACHE, n), k) for n in ("A", "B")]
theta = Theta(noise=NOISE, L=L_SHAPE, eps=EPS, eta=10 * NOISE, zeta=100 * NOISE, ell=2.0, log_kappa_mean=36.0,
              log_kappa_std=2.0, gain_db=GAIN_DB)
r = np.asarray(bin_ranges(passes[0].n_bins, passes[0].dr))
bands = [(0, 15), (15, 25), (25, 35), (35, 60)]
log("share of bins labelled lit, by side and range band (m): " + ", ".join(f"{lo}-{hi}" for lo, hi in bands))
base = None
for state in ("start", "k27", "k9", ""):
    path = RUNS / ("fr2.npz" if state == "" else f"fr2_{state}.npz")
    if not path.exists():
        continue
    extra = load_depths(path)[2]
    psi = extra["psi"] if "psi" in extra else psi_of(theta)
    th = coarsen_theta(with_psi(theta, psi), k)
    lab = [labels_of(p, th) for p in passes]
    if base is None:
        base = lab
    for c, sd in enumerate(("port", "stbd")):
        lit = np.concatenate([x[:, c] for x in lab])
        was = np.concatenate([x[:, c] for x in base])
        shares = [lit[:, (r >= lo) & (r < hi)].mean() for lo, hi in bands]
        log(f"  {state or 'final':6s} {sd}: g1 {psi[c, 2]:.3f} dB/m  " + "  ".join(f"{x:6.1%}" for x in shares)
            + f"   bins lit at the start now dark: {np.mean(was & ~lit) / max(np.mean(was), 1e-12):.1%}")
