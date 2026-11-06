#!/usr/bin/env bash
# Per-kernel times of CloverLeaf (2D or 3D), AOT against the JIT without fusion (every loop is its own group, so the times are
# exact) and with fusion, from OPS's own timing table (deck keyword profiler_on, OPS_DIAGS=2). Run through a SLURM job (source scripts/env.sh first).
# Output: results/cpu-kernels-<job>/{aot,jit-nofuse,jit}.txt (the tables) and kernels.txt (joined, sorted by AOT time).
#
# Environment: as cpu.sh, plus DIM (2 or 3, default 3), STEPS (default 20).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

AOT=${OPS_AOT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-apps-new}
JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
THREADS=${THREADS:-$(nproc)}
STEPS=${STEPS:-20}
DIM=${DIM:-3}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/cpu-kernels-${OPSJIT_JOB:-local}
mkdir -p "$OUT"
export OMP_NUM_THREADS=$THREADS
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf}
mkdir -p "$OPS_JIT_CACHE_DIR"

if [ "$DIM" = 3 ]; then
  C=$OPS_SRC/apps/c/CloverLeaf_3D; deck=clover_bm1s_short.in
  aot_exe=$AOT/apps/c/CloverLeaf_3D/clover_leaf_3D_openmp; jit_exe=$JIT/clover_leaf_3D_jit
else
  C=$OPS_SRC/apps/c/CloverLeaf; deck=clover_bm16_short.in
  aot_exe=$AOT/apps/c/CloverLeaf/clover_leaf_openmp; jit_exe=$JIT/clover_leaf_jit
fi
work=$OPSJIT_WORK/perf-run/kernels$DIM
rm -rf "$work"; mkdir -p "$work"
{ sed "s/end_step=.*/end_step=$STEPS/" "$C/$deck"; echo " profiler_on=1"; } >"$work/clover.in"
# The deck ends with *endclover: the keyword has to be inside it.
sed -i '/^\*endclover/d' "$work/clover.in"; echo "*endclover" >>"$work/clover.in"

run() { # run <label> <command...>
  local label=$1; shift
  (cd "$work" && rm -f prof.out && "$@" -OPS_DIAGS=2 >"$OUT/$label.out" 2>&1; cp prof.out "$OUT/$label.prof" 2>/dev/null)
  # OPS's table: header "Name  Count  Time  ..." then one line per kernel
  awk '/^Name +Count/ {t=1; next} t && NF>=3 {print}' "$OUT/$label.prof" >"$OUT/$label.txt"
}
run aot "$aot_exe"
(cd "$work" && OPS_JIT_FUSION=1 "$jit_exe" -OPS_DIAGS=2 >"$OUT/jit-warm.out" 2>&1)
OPS_JIT_FUSION=0 run jit-nofuse "$jit_exe"
OPS_JIT_FUSION=1 run jit "$jit_exe"
# join by kernel name: name, AOT time, JIT-nofuse time
awk 'FNR==NR {a[$1]=$3; next} ($1 in a) {printf "%-40s %10s %10s\n", $1, a[$1], $3}' "$OUT/aot.txt" "$OUT/jit-nofuse.txt" | sort -k2 -n -r >"$OUT/kernels.txt"
head -12 "$OUT/aot.prof"; head -30 "$OUT/kernels.txt"
echo "results: $OUT"
