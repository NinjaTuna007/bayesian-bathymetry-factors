#!/usr/bin/env bash
# Run a diagnostic script pinned to cores 0-5, e.g.: scripts/optdiag/run.sh t0_setup.py 27 start
cd "$(dirname "$0")/../.." || exit 1
export XLA_FLAGS="--xla_force_host_platform_device_count=1" OMP_NUM_THREADS=6 MPLBACKEND=Agg
export PYTHONPATH=.:scripts:scripts/optdiag
script=$1
shift
exec taskset -c 0-5 nix develop -c python "scripts/optdiag/$script" "$@"
