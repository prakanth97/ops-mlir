#!/usr/bin/env bash
# Per-kernel times of CloverLeaf 2D, AOT against the JIT, on one CPU node (OPS_DIAGS=2 prints OPS's timing table).
# Run through a SLURM job (source scripts/env.sh first). Output: results/cpu-diag-<job>/{aot,jit,jit-nofuse}.out
#
# Environment: as cpu.sh, plus STEPS (default 10), DECK (default clover_bm16_short.in).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

AOT=${OPS_AOT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-apps-new}
JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
THREADS=${THREADS:-$(nproc)}
STEPS=${STEPS:-10}
DECK=${DECK:-clover_bm16_short.in}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/cpu-diag-${OPSJIT_JOB:-local}
mkdir -p "$OUT"
export OMP_NUM_THREADS=$THREADS OPS_DIAGS=2
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf}
mkdir -p "$OPS_JIT_CACHE_DIR"

C=$OPS_SRC/apps/c/CloverLeaf
work=$OPSJIT_WORK/perf-run/diag
rm -rf "$work"; mkdir -p "$work/aot" "$work/jit"
for d in aot jit; do
  sed "s/end_step=.*/end_step=$STEPS/" "$C/$DECK" >"$work/$d/clover.in"
done
(cd "$work/aot" && time "$AOT/apps/c/CloverLeaf/clover_leaf_openmp" >"$OUT/aot.out" 2>&1)
(cd "$work/jit" && OPS_JIT_FUSION=1 "$JIT/clover_leaf_jit" >"$OUT/jit-warm.out" 2>&1)
(cd "$work/jit" && time OPS_JIT_FUSION=1 OPS_JIT_STATS=1 "$JIT/clover_leaf_jit" >"$OUT/jit.out" 2>&1)
(cd "$work/jit" && time OPS_JIT_FUSION=0 OPS_JIT_STATS=1 "$JIT/clover_leaf_jit" >"$OUT/jit-nofuse.out" 2>&1)
echo "results: $OUT"
