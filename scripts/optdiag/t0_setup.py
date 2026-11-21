"""Build and cache a level set-up the diagnostics use; time one evaluation and the Jacobians."""
import sys
import time

from common import log, setup

from bbf.inference.objective import data_term, value_and_grad

k, state = int(sys.argv[1]), sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
s = setup(k, state, every)
t = time.time()
ell, _, det = value_and_grad(s.problem, s.d, s.beta, data_term(s.problem), psi=s.psi, xi=s.xi)
log(f"k={k} {state}: ell {ell:.1f} data {det['data']:.1f} fr {det['first_return_value']:.1f} xi sd {s.xi.std():.2f} "
    f"({time.time() - t:.0f}s)")
t = time.time()
s.jac()
log(f"jacobians {time.time() - t:.0f}s")
