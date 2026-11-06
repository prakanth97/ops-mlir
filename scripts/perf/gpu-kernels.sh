#!/usr/bin/env bash
# Per-kernel times of CloverLeaf (2D or 3D) on a GPU, OPS AOT CUDA against the JIT without fusion (OPS_JIT_SYNC=1, so every group
# is timed to the end of its kernels), from OPS's timing table (deck keyword profiler_on, -OPS_DIAGS=2). Run through
# a SLURM job (source scripts/env.sh first) on a GPU partition. Output: results/gpu-kernels-<job>/{aot,jit-nofuse}.txt and kernels.txt (joined, sorted by AOT
# time: name, AOT [s], JIT [s]).
#
# Environment: as gpu.sh, plus DIM (2 or 3, default 3), STEPS (default 20).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

AOT=${OPS_AOT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-apps-new}
JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
STEPS=${STEPS:-20}
DIM=${DIM:-3}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/gpu-kernels-${OPSJIT_JOB:-local}
mkdir -p "$OUT"
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf-gpu-kernels}
mkdir -p "$OPS_JIT_CACHE_DIR"

if [ "$DIM" = 3 ]; then
  C=$OPS_SRC/apps/c/CloverLeaf_3D; deck=clover_bm1s_short.in
  aot_exe=$AOT/apps/c/CloverLeaf_3D/clover_leaf_3D_cuda; jit_exe=$JIT/clover_leaf_3D_jit_cuda
else
  C=$OPS_SRC/apps/c/CloverLeaf; deck=clover_bm16_short.in
  aot_exe=$AOT/apps/c/CloverLeaf/clover_leaf_cuda; jit_exe=$JIT/clover_leaf_jit_cuda
fi
work=$OPSJIT_WORK/perf-run-gpu/kernels$DIM
rm -rf "$work"; mkdir -p "$work"
sed "s/end_step=.*/end_step=$STEPS/" "$C/$deck" | sed '/^\*endclover/d' >"$work/clover.in"
{ echo " profiler_on=1"; echo "*endclover"; } >>"$work/clover.in"

run() { # run <label> <command...>
  local label=$1; shift
  (cd "$work" && rm -f prof.out && "$@" -OPS_DIAGS=2 >"$OUT/$label.out" 2>&1; cp prof.out "$OUT/$label.prof" 2>/dev/null)
  awk '/^Name +Count/ {t=1; next} t && NF>=3 {print}' "$OUT/$label.prof" >"$OUT/$label.txt"
}
run aot "$aot_exe" OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 OPS_BLOCK_SIZE_Z=1
(cd "$work" && OPS_JIT_FUSION=0 "$jit_exe" >"$OUT/jit-warm.out" 2>&1)
OPS_JIT_FUSION=0 OPS_JIT_SYNC=1 run jit-nofuse "$jit_exe"
awk 'FNR==NR {a[$1]=$3; next} ($1 in a) {printf "%-40s %10s %10s\n", $1, a[$1], $3}' "$OUT/aot.txt" "$OUT/jit-nofuse.txt" | sort -k2 -n -r >"$OUT/kernels.txt"
head -40 "$OUT/kernels.txt"
awk '{a+=$2; b+=$3} END {print "sum", a, b}' "$OUT/kernels.txt"
echo "results: $OUT"
