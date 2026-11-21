r"""One table over the fits of `t2_fit.py`: every end state scored the same way.

Usage: run.sh t8_summary.py K STATE NAME [NAME ...]

Every state is scored by the untempered posterior terms with the log-kappa modes found from a grid of
starts (so the score is a function of the state), next to the first-return residuals, the size of the
move and the step records (actual and predicted gain).
"""
import dataclasses
import pickle
import sys

import numpy as np
from common import OUT, log, setup

from bbf.inference.objective import data_term, value_and_grad
from bbf.ping.first_return import _batched

k, state, names = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
s = setup(k, state)
pb = dataclasses.replace(s.problem, mode_grid=49)
F = data_term(pb)


def score(d, beta, psi, xi):
    _, _, det = value_and_grad(pb, d, beta, F, psi=psi, xi=xi)
    fr = s.first_return
    res = []
    for P, r0 in zip(pb.poses, fr.r0):
        rt = np.stack([np.asarray(_batched(pb.mesh, c, fr.tau)(np.asarray(d), np.asarray(P), np.asarray(fr.u))[0])
                       for c in (0, 1)], axis=1)
        res.append(r0 - rt)
    return {"data": det["data"], "fr": det["first_return_value"], "prior_d": pb.prior.log_density(d, beta),
            "prior_xi": pb.xi_log_prior(xi)[0], "prior_psi": pb.psi_log_prior(psi)[0],
            "res": [np.nanmedian(r[:, c]) for r in res for c in (0, 1)],
            "in": [np.nanmean(np.abs(r[:, c][np.isfinite(r[:, c])]) < 0.3) for r in res for c in (0, 1)]}


base = score(s.d, s.beta, s.psi, s.xi)
log(f"start: data {base['data']:.0f}, first return {base['fr']:.0f}, prior d {base['prior_d']:.0f}, xi {base['prior_xi']:.0f}; "
    f"first-return residual median (A port, A stbd, B port, B stbd) {np.round(base['res'], 2)} m, within 0.3 m "
    f"{np.round(base['in'], 2)}")
log("name: change of data / first return / prior d / prior xi / prior psi; sd and max of the depth move; xi sd; "
    "psi (centre, width deg; g1); first-return medians; within 0.3 m")
out = {}
for nm in names:
    f = pickle.loads((OUT / f"fit_{nm}.pkl").read_bytes())
    sc = score(f["d"], f["beta"], f["psi"], f["xi"])
    out[nm] = sc
    dd = f["d"] - s.d
    psi = f["psi"]
    log(f"{nm:10s} {f['opt']}")
    log(f"    data {sc['data'] - base['data']:+9.0f}  fr {sc['fr'] - base['fr']:+7.0f}  prior d {sc['prior_d'] - base['prior_d']:+7.0f}  "
        f"xi {sc['prior_xi'] - base['prior_xi']:+7.0f}  psi {sc['prior_psi'] - base['prior_psi']:+5.1f} | move sd {dd.std():.2f} "
        f"max {np.abs(dd).max():.2f} m | xi sd {f['xi'].std():.2f} | beam {np.degrees(psi[:, 0]).round(1)}/"
        f"{np.degrees(psi[:, 1]).round(1)} g1 {psi[:, 2].round(3)} | res {np.round(sc['res'], 2)} in {np.round(sc['in'], 2)}")
    for i, r in enumerate(f["recs"][1:]):
        log(f"      step {i}: lam {r['lam']:.1e} actual {r['actual']:+10.1f} predicted {r['pred']:+10.1f} ratio {r['rho']:5.2f} "
            f"shorten {r['over']:7.1f} scale {r['scale']:5.1f} clipped {r['n_clip']:6d} raw max {r['raw_dd_max']:7.1f} m "
            f"cold-warm {r.get('cold_minus_warm', np.nan):+8.0f}")
(OUT / f"summary_k{k}_{state}.pkl").write_bytes(pickle.dumps({"base": base, "fits": out}))
