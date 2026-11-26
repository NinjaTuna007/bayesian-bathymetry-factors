#!/usr/bin/env bash
# The second batch at k = 27 from the start state: step control, tempering and the first-return weight on top
# of the consistent objective (log-kappa modes from a grid, shadow factor differentiated through log-kappa).
cd "$(dirname "$0")/../.." || exit 1
R=scripts/optdiag/run.sh
O=runs/optdiag
C="psi=fixed grid=49 shadow_stop=0"
LM="ratio=1 floor=0.5 clip=scale relin=1"
$R t9_path.py 27 > $O/t9_path.log 2>&1
$R t2_fit.py lm 27 start 1 6 $C $LM > $O/fit_lm.log 2>&1
$R t2_fit.py t02lm 27 start 1 6 $C $LM temper=0.02 > $O/fit_t02lm.log 2>&1
$R t1_state.py 9 k27 > $O/t1_9_k27.log 2>&1
$R t1_vertices.py 9 k27 > $O/t1v_9_k27.log 2>&1
$R t2_fit.py t02lmfr 27 start 1 6 $C $LM temper=0.02 fr_sigma=1.0 > $O/fit_t02lmfr.log 2>&1
$R t2_fit.py t02 27 start 1 6 $C temper=0.02 > $O/fit_t02.log 2>&1
echo done > $O/queue2.done
