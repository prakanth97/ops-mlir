#!/usr/bin/env bash
# CPU performance: OPS AOT (OpenMP) against the JIT, on one node. Run through a SLURM job (source scripts/env.sh first).
#
# For every (application, deck): the AOT executable of the OPS build (OPS_AOT_BUILD), then the JIT executable with
#   jit        the default (fusion on)
#   jit-nofuse OPS_JIT_FUSION=0
# each with the same number of threads (pinned: OMP_PROC_BIND, OMP_PLACES from env.sh). The JIT runs once to fill the
# disk cache (not timed), then NRUNS timed runs; the best time is reported. Output: one JSON object per line.
#
# Environment: OPS_AOT_BUILD (default $OPSJIT_BUILD_ROOT/OPS-apps-new), JIT_BUILD (default $OPSJIT_BUILD_ROOT/OPS-jit-apps),
# THREADS (default: all cores of the node), NRUNS (default 3), OPSJIT_RESULTS, CASES (regex of the cases to run), SKIP_AOT=1.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

AOT=${OPS_AOT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-apps-new}
JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
TGV=${TGV_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-tgv}
THREADS=${THREADS:-$(nproc)}
NRUNS=${NRUNS:-3}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/cpu-${OPSJIT_JOB:-local}.json
mkdir -p "$(dirname "$OUT")"
: >"$OUT"
export OMP_NUM_THREADS=$THREADS
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf}
mkdir -p "$OPS_JIT_CACHE_DIR"

now() { date +%s.%N; }

# timed <label> <dir> <command...>: best of NRUNS wall times; prints the JSON line
timed() {
  local label=$1 dir=$2; shift 2
  local best=""
  for _ in $(seq "$NRUNS"); do
    local t0 t1 dt
    t0=$(now)
    (cd "$dir" && "$@" >run.out 2>&1) || { echo "run failed: $label" >&2; tail -3 "$dir/run.out" >&2; return 1; }
    t1=$(now)
    dt=$(echo "$t1 - $t0" | bc -l)
    if [ -z "$best" ] || [ "$(echo "$dt < $best" | bc -l)" = 1 ]; then best=$dt; fi
  done
  printf '{"case": "%s", "threads": %s, "seconds": %.3f}\n' "$label" "$THREADS" "$best" | tee -a "$OUT"
}

# perf_case <name> <aot exe> <aot dir> <jit exe> <jit dir> <deck> <deck dir>
perf_case() {
  local name=$1 aot_exe=$2 aot_dir=$3 jit_exe=$4 jit_dir=$5 deck=$6 deck_dir=$7
  [[ "$name" =~ ${CASES:-.*} ]] || return 0
  [ "${SKIP_AOT:-0}" = 1 ] && aot_exe=/nonexistent
  local work=$OPSJIT_WORK/perf-run/$name
  mkdir -p "$work/aot" "$work/jit"
  cp "$aot_dir"/*.in "$work/aot/" 2>/dev/null || true
  cp "$jit_dir"/*.in "$work/jit/" 2>/dev/null || true
  if [ -n "$deck" ]; then
    cp "$deck_dir/$deck" "$work/aot/clover.in"
    cp "$deck_dir/$deck" "$work/jit/clover.in"
  fi
  [ -f "$aot_exe" ] && timed "$name/aot" "$work/aot" "$aot_exe"
  if [ -f "$jit_exe" ]; then
    (cd "$work/jit" && OPS_JIT_FUSION=1 "$jit_exe" >warm.out 2>&1)
    OPS_JIT_FUSION=1 timed "$name/jit" "$work/jit" "$jit_exe"
    OPS_JIT_FUSION=0 timed "$name/jit-nofuse" "$work/jit" "$jit_exe"
  fi
}

C=$OPS_SRC/apps/c
perf_case clover2d-bm16short "$AOT/apps/c/CloverLeaf/clover_leaf_openmp" "$AOT/apps/c/CloverLeaf/tmp" \
          "$JIT/clover_leaf_jit" "$JIT/clover_leaf_tmp" clover_bm16_short.in "$C/CloverLeaf"
perf_case clover3d-bm1sshort "$AOT/apps/c/CloverLeaf_3D/clover_leaf_3D_openmp" "$AOT/apps/c/CloverLeaf_3D/tmp" \
          "$JIT/clover_leaf_3D_jit" "$JIT/clover_leaf_3D_tmp" clover_bm1s_short.in "$C/CloverLeaf_3D"
for p in DP SP SPDP; do
  perf_case tgv-$p "$TGV/tgv_${p}_aot" "$TGV/tgv_${p}_aot_tmp" "$TGV/tgv_${p}_jit" "$TGV/tgv_${p}_tmp" "" ""
done
echo "results: $OUT"
