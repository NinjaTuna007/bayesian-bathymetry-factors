#!/usr/bin/env bash
# The third batch at k = 27 from the start state: the trust-region step with and without tempering.
cd "$(dirname "$0")/../.." || exit 1
R=scripts/optdiag/run.sh
O=runs/optdiag
C="psi=fixed grid=49 shadow_stop=0"
TR="ratio=1 floor=0.5 clip=trust relin=1"
$R t2_fit.py tr 27 start 1 6 $C $TR > $O/fit_tr.log 2>&1
$R t2_fit.py t02tr 27 start 1 6 $C $TR temper=0.02 > $O/fit_t02tr.log 2>&1
$R t1_state.py 9 k27 > $O/t1_9_k27.log 2>&1
$R t1_vertices.py 9 k27 > $O/t1v_9_k27.log 2>&1
$R t2_fit.py t02trfr 27 start 1 6 $C $TR temper=0.02 fr_sigma=1.0 > $O/fit_t02trfr.log 2>&1
$R t2_fit.py t2tr 27 start 1 6 $C $TR temper=0.2 > $O/fit_t2tr.log 2>&1
echo done > $O/queue3.done
