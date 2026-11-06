#!/usr/bin/env bash
# Time per group of the JIT (OPS_JIT_TRACE=1), summed over the run: which kernels (or fused groups) take the time. Run through
# a SLURM job (source scripts/env.sh first). Output: results/cpu-groups-<job>/{nofuse,fuse}.txt = total seconds, calls, members.
#
# Environment: as cpu.sh, plus DIM (2 or 3, default 3), STEPS (default 20).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
THREADS=${THREADS:-$(nproc)}
STEPS=${STEPS:-20}
DIM=${DIM:-3}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/cpu-groups-${OPSJIT_JOB:-local}
mkdir -p "$OUT"
export OMP_NUM_THREADS=$THREADS
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf}
mkdir -p "$OPS_JIT_CACHE_DIR"

if [ "$DIM" = 3 ]; then
  C=$OPS_SRC/apps/c/CloverLeaf_3D; deck=clover_bm1s_short.in; exe=$JIT/clover_leaf_3D_jit
else
  C=$OPS_SRC/apps/c/CloverLeaf; deck=clover_bm16_short.in; exe=$JIT/clover_leaf_jit
fi
work=$OPSJIT_WORK/perf-run/groups$DIM
rm -rf "$work"; mkdir -p "$work"
sed "s/end_step=.*/end_step=$STEPS/" "$C/$deck" >"$work/clover.in"
cd "$work"
OPS_JIT_FUSION=1 "$exe" >warm.out 2>&1

summarise() { # trace file -> total seconds, calls, members
  awk '/^ops_jit: running / {sub(/^ops_jit: running [^:]*:/, ""); names=$0; next}
       /^ops_jit:   [0-9.]+ s$/ {t[names]+=$2; n[names]++; total+=$2}
       END {for (k in t) printf "%10.4f %7d %s\n", t[k], n[k], k; printf "%10.4f total\n", total}' "$1" | sort -rn
}
for mode in nofuse fuse; do
  f=1; [ "$mode" = nofuse ] && f=0
  OPS_JIT_FUSION=$f OPS_JIT_TRACE=1 "$exe" >"$OUT/$mode.trace" 2>&1
  summarise "$OUT/$mode.trace" >"$OUT/$mode.txt"
  echo "== $mode"; head -25 "$OUT/$mode.txt" | cut -c1-200
done
echo "results: $OUT"
for f in 1 0; do
  echo "== statistics, OPS_JIT_FUSION=$f"
  OPS_JIT_FUSION=$f OPS_JIT_STATS=1 "$exe" >"$OUT/stats$f.out" 2>&1
  grep -a -E "counters|time \[|Wall" "$OUT/stats$f.out" | cut -c1-300
done
