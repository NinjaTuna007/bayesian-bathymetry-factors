r"""How overconfident is the intensity term: dispersion and correlation of the residuals at a state.

Usage: run.sh t3_corr.py K STATE [FIT.pkl]

With :math:`z_b = y_b / \bar I_b - 1` on the scored bins, the gamma model has :math:`\mathrm{Var}\, z = 1/(kL)`.
Reports the measured variance (robustly, from the central quantiles, since spikes are a separate
component), the implied looks, the lag-1 correlations along range and across pings, and the factor
:math:`\tau` that discounts the log likelihood for a smooth parameter,
:math:`\tau = \frac{1}{kL\,\mathrm{Var}\,z} \cdot \frac{1 - \rho_r}{1 + \rho_r} \cdot \frac{1 - \rho_t}{1 + \rho_t}`.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from common import log, setup

from bbf.inference.objective import data_term, value_and_grad
from bbf.ping.factor import predicted
from bbf.ping.theta import with_psi

k, state = int(sys.argv[1]), sys.argv[2]
s = setup(k, state)
pb = s.problem
d, beta, psi, xi = s.d, s.beta, s.psi, s.xi
if len(sys.argv) > 3:
    f = pickle.loads(Path(sys.argv[3]).read_bytes())
    d, beta, psi, xi = f["d"], f["beta"], f["psi"], f["xi"]
_, _, det = value_and_grad(pb, d, beta, data_term(pb), psi=psi, xi=xi)
th = with_psi(pb.theta, psi)
kL = th.L
for p, nm in enumerate("AB"):
    for c, sd in enumerate(("port", "stbd")):
        y = pb.y[p][:, c]
        use = pb.uses()[p][:, c] & (y > 0)
        Ib = np.asarray(predicted(det["S"][p][:, c], det["u"][p][:, c, None], th, c, pb.dr)[0])
        z = np.where(use, np.log(np.where(use, y / Ib, 1.0)), np.nan)  # log residual: robust to the spikes
        zc = z - np.nanmedian(z)
        # the log of a Gam(kL, mean 1) variate has variance trigamma(kL), about 1/kL
        q16, q84 = np.nanpercentile(z, [16, 84])
        var = ((q84 - q16) / 2) ** 2
        lim = 3 * np.sqrt(var)
        zc = np.where(np.abs(zc) < lim, zc, np.nan)

        def corr(a, b):
            ok = np.isfinite(a) & np.isfinite(b)
            return float(np.corrcoef(a[ok], b[ok])[0, 1])

        rr, rt = corr(zc[:, :-1], zc[:, 1:]), corr(zc[:-1], zc[1:])
        tau_pt = 1.0 / (kL * var)
        tau = tau_pt * (1 - rr) / (1 + rr) * (1 - rt) / (1 + rt)
        log(f"{nm} {sd}: {int(use.sum())} bins, sd of log residual {np.sqrt(var):.2f} (model {1 / np.sqrt(kL):.2f}), "
            f"looks implied {1 / var:.1f} of {kL:.1f}; lag-1 corr range {rr:.2f}, pings {rt:.2f}; "
            f"tau pointwise {tau_pt:.2f}, for smooth parameters {tau:.4f}")
