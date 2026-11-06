#!/usr/bin/env bash
# CPU: the effect of the largest fused group (OPS_JIT_FUSION_MAX) on the TGV and CloverLeaf runs. Run through
# a SLURM job (source scripts/env.sh first). Output: one JSON object per line in results/cpu-fusion-sweep-<job>.json.
#
# Environment: as cpu.sh, plus SIZES (default "1 2 4 8 16 32"; 1 = no fusion), CASES (regex of tgv-DP, tgv-SP, clover3d, default all of them).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
TGV=${TGV_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-tgv}
THREADS=${THREADS:-$(nproc)}
SIZES=${SIZES:-"1 2 4 8 16 32"}
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/cpu-fusion-sweep-${OPSJIT_JOB:-local}.json
mkdir -p "$(dirname "$OUT")"
: >"$OUT"
export OMP_NUM_THREADS=$THREADS
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf-fusion}
mkdir -p "$OPS_JIT_CACHE_DIR"
now() { date +%s.%N; }

sweep() { # sweep <name> <exe> <dir> [<deck> <deck dir>]
  local name=$1 exe=$2 dir=$3 deck=${4:-} deck_dir=${5:-}
  [[ "$name" =~ ${CASES:-.*} ]] || return 0
  local work=$OPSJIT_WORK/perf-run/fusion-$name
  rm -rf "$work"; mkdir -p "$work"
  cp "$dir"/*.in "$work/" 2>/dev/null || true
  [ -n "$deck" ] && cp "$deck_dir/$deck" "$work/clover.in"
  for m in $SIZES; do
    local fusion=1 max=$m
    [ "$m" = 1 ] && fusion=0
    (cd "$work" && OPS_JIT_FUSION=$fusion OPS_JIT_FUSION_MAX=$max "$exe" >warm.out 2>&1)
    local t0 dt
    t0=$(now)
    (cd "$work" && OPS_JIT_FUSION=$fusion OPS_JIT_FUSION_MAX=$max "$exe" >run.out 2>&1)
    dt=$(echo "$(now) - $t0" | bc -l)
    printf '{"case": "%s", "fusion_max": %s, "threads": %s, "seconds": %.3f}\n' "$name" "$m" "$THREADS" "$dt" | tee -a "$OUT"
  done
}

C=$OPS_SRC/apps/c
sweep clover3d "$JIT/clover_leaf_3D_jit" "$JIT/clover_leaf_3D_tmp" clover_bm1s_short.in "$C/CloverLeaf_3D"
sweep tgv-SP "$TGV/tgv_SP_jit" "$TGV/tgv_SP_tmp"
sweep tgv-DP "$TGV/tgv_DP_jit" "$TGV/tgv_DP_tmp"
echo "results: $OUT"
