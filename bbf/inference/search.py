r"""Global search over the large-scale trend, to cross between sharp local modes.

Range timing makes the depth posterior sharply multimodal at fine scale, while the data still
rank the large-scale shape clearly: started at the true trend, Gauss--Newton refines to a mode
hundreds of log units above the one reached from a tilted start, but it cannot get from one to
the other. The trend :math:`\beta` (two slopes and an intercept) is the low-dimensional direction
between them, so each candidate trend is tried directly: the depths are moved by the change in
trend, :math:`d' = d + X(\beta' - \beta)`, keeping the fine-scale residual and the mean depth over
the observed vertices; a few Gauss--Newton steps refine them; and the candidate with the
highest refined log posterior wins.
"""
import itertools
import time

import numpy as np

from bbf.inference.gauss_newton import gauss_newton
from bbf.inference.loop import jacobians
from bbf.inference.objective import Problem, data_term, value


def search_trend(problem: Problem, d, beta, slopes_x, slopes_y=(0.0,), observed=None, steps: int = 4,
                 log=print) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    r"""Best (d, :math:`\beta`) over trend candidates with absolute slopes: every pair of slopes_x and slopes_y.

    observed: vertices whose mean depth each candidate keeps (default all); steps: Gauss--Newton
    steps of refinement per candidate. Returns the winner and a record per candidate.
    """
    d, beta = np.asarray(d, dtype=float), np.asarray(beta, dtype=float)
    if problem.prior.n_beta != 3:
        raise ValueError("the trend search needs a planar trend")
    X = problem.prior.X
    obs = np.ones(len(d), bool) if observed is None else np.asarray(observed, bool)
    F = data_term(problem)
    best, records = None, []
    for sx, sy in itertools.product(slopes_x, slopes_y):
        t0 = time.time()
        b = beta.copy()
        b[0], b[1] = sx, sy
        b[2] = beta[2] + np.mean(X[obs, :2] @ (beta[:2] - b[:2]))  # keep the mean depth over obs
        dc = d + X @ (b - beta)
        dc, bc, _, _ = gauss_newton(problem, dc, b, jacobians(problem, dc), max_iter=steps, log=lambda _: None)
        ell = value(problem, dc, bc, F)
        records.append({"slopes": (sx, sy), "ell": ell, "time": time.time() - t0})
        log(f"  trend slopes ({sx:+.3f}, {sy:+.3f}): refined ell {ell:.1f} ({time.time() - t0:.0f}s)")
        if best is None or ell > best[0]:
            best = (ell, dc, bc)
    assert best is not None
    return best[1], best[2], records
