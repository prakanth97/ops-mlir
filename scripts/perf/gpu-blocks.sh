#!/usr/bin/env bash
# GPU thread block sweep of the JIT (OPS_JIT_GPU_BLOCK) on CloverLeaf 2D/3D and TGV single precision. Run through
# a SLURM job (source scripts/env.sh first) on a GPU partition. Output: one JSON object per line in results/gpu-blocks-<job>.json.
#
# Environment: as gpu.sh, plus BLOCKS (space separated x,y,z, default a list of 1-D, 2-D and 3-D shapes of 128 to 512 threads),
# NRUNS (default 2).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
TGV=${TGV_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-tgv}
NRUNS=${NRUNS:-2}
BLOCKS=${BLOCKS:-"32,4,1 64,2,1 128,1,1 256,1,1 32,8,1 16,16,1 32,4,2 32,2,4 64,4,1 128,2,1 16,4,4 8,8,4"}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/gpu-blocks-${OPSJIT_JOB:-local}.json
mkdir -p "$(dirname "$OUT")"
: >"$OUT"
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf-gpu-blocks}
mkdir -p "$OPS_JIT_CACHE_DIR"
now() { date +%s.%N; }

# sweep <name> <jit exe> <jit dir> [<deck> <deck dir>]
sweep() {
  local name=$1 exe=$2 dir=$3 deck=${4:-} deck_dir=${5:-}
  local work=$OPSJIT_WORK/perf-run-gpu/blocks-$name
  rm -rf "$work"; mkdir -p "$work"
  cp "$dir"/*.in "$work/" 2>/dev/null || true
  [ -n "$deck" ] && cp "$deck_dir/$deck" "$work/clover.in"
  for b in $BLOCKS; do
    (cd "$work" && OPS_JIT_GPU_BLOCK=$b "$exe" >warm.out 2>&1) || { echo "{\"case\": \"$name\", \"block\": \"$b\", \"seconds\": null}" | tee -a "$OUT"; continue; }
    local best=""
    for _ in $(seq "$NRUNS"); do
      local t0 dt
      t0=$(now)
      (cd "$work" && OPS_JIT_GPU_BLOCK=$b "$exe" >run.out 2>&1)
      dt=$(echo "$(now) - $t0" | bc -l)
      if [ -z "$best" ] || [ "$(echo "$dt < $best" | bc -l)" = 1 ]; then best=$dt; fi
    done
    printf '{"case": "%s", "block": "%s", "gpu": "%s", "seconds": %.3f}\n' "$name" "$b" "$GPU" "$best" | tee -a "$OUT"
  done
}

C=$OPS_SRC/apps/c
sweep clover2d "$JIT/clover_leaf_jit_cuda" "$JIT/clover_leaf_tmp" clover_bm16_short.in "$C/CloverLeaf"
sweep clover3d "$JIT/clover_leaf_3D_jit_cuda" "$JIT/clover_leaf_3D_tmp" clover_bm1s_short.in "$C/CloverLeaf_3D"
sweep tgv-SP "$TGV/tgv_SP_jit_cuda" "$TGV/tgv_SP_tmp"
echo "results: $OUT"
