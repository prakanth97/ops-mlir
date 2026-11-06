#!/usr/bin/env bash
# Poor man's profiler: runs CloverLeaf 2D (JIT) and samples the stack of the main thread with gdb once a second.
# Run through a SLURM job (source scripts/env.sh first). Output: results/sample-<job>.txt (the 12 innermost frames of each sample).
#
# Environment: as cpu.sh, plus STEPS (default 87), DECK (default clover_bm16_short.in).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
DIM=${DIM:-2}
THREADS=${THREADS:-$(nproc)}
STEPS=${STEPS:-87}
DECK=${DECK:-$([ "$DIM" = 3 ] && echo clover_bm1s_short.in || echo clover_bm16_short.in)}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/sample-${OPSJIT_JOB:-local}.txt
export OMP_NUM_THREADS=$THREADS
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf}
work=$OPSJIT_WORK/perf-run/sample
rm -rf "$work"; mkdir -p "$work"
APP=$([ "$DIM" = 3 ] && echo CloverLeaf_3D || echo CloverLeaf); EXE=$([ "$DIM" = 3 ] && echo clover_leaf_3D_jit || echo clover_leaf_jit)
sed "s/end_step=.*/end_step=$STEPS/" "$OPS_SRC/apps/c/$APP/$DECK" >"$work/clover.in"
cd "$work"
"$JIT/$EXE" >run.out 2>&1 &
pid=$!
sleep ${DELAY:-8} # past the start-up
: >"$OUT"
while kill -0 "$pid" 2>/dev/null; do
  gdb -p "$pid" -batch -ex "thread 1" -ex "bt 12" 2>/dev/null | grep -E '^#' | cut -c1-170 >>"$OUT"
  echo "----" >>"$OUT"
  sleep ${PERIOD:-1}
done
wait "$pid"
echo "results: $OUT"
