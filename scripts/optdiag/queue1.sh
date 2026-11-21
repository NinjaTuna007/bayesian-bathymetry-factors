#!/usr/bin/env bash
# The first batch of experiments at k = 27 from the start state of run fr2, one after the other.
cd "$(dirname "$0")/../.." || exit 1
R=scripts/optdiag/run.sh
O=runs/optdiag
$R t2_fit.py nocap 27 start 1 3 clip=none shorten=0 > $O/fit_nocap.log 2>&1
$R t2_fit.py psifix 27 start 1 6 psi=fixed > $O/fit_psifix.log 2>&1
$R t4_fd.py 27 start 49 0 > $O/t4_27_start_grid.log 2>&1
$R t2_fit.py grid 27 start 1 6 psi=fixed grid=49 shadow_stop=0 > $O/fit_grid.log 2>&1
$R t2_fit.py gridpsi 27 start 1 6 grid=49 shadow_stop=0 > $O/fit_gridpsi.log 2>&1
echo done > $O/queue1.done
