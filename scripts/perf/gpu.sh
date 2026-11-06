#!/usr/bin/env bash
# GPU performance: OPS AOT CUDA against the JIT on one GPU. Run through a SLURM job (source scripts/env.sh first) on a GPU partition:
#   sbatch -p gpu --gres=gpu:a100:1 -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log --wrap 'source scripts/env.sh && scripts/perf/gpu.sh'
#
# CloverLeaf 2D/3D: the AOT executable of the OPS build (OPS_AOT_BUILD) against the JIT (default, and no fusion). TGV
# (double, single, mixed): the AOT baseline built by apps/ (`tgv_<p>_aot_cuda`) against `tgv_<p>_jit_cuda`, in TGV_BUILD.
# The JIT runs once first (not timed) to fill the disk cache; NRUNS timed runs, the best is reported (wall time of the
# whole run, which includes the start-up). Output: one JSON object per line in $OPSJIT_RESULTS/gpu-<job>.json.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/../env.sh"

AOT=${OPS_AOT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-apps-new}
JIT=${JIT_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
TGV=${TGV_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-tgv}
NRUNS=${NRUNS:-3}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
OUT=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}/gpu-${OPSJIT_JOB:-local}.json
mkdir -p "$(dirname "$OUT")"
: >"$OUT"
export OPS_JIT_CACHE_DIR=${OPS_JIT_CACHE_DIR:-$OPSJIT_WORK/cache/perf-gpu}
mkdir -p "$OPS_JIT_CACHE_DIR"

now() { date +%s.%N; }

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
  printf '{"case": "%s", "gpu": "%s", "seconds": %.3f}\n' "$label" "$GPU" "$best" | tee -a "$OUT"
}

# perf_case <name> <aot exe> <aot dir> <jit exe> <jit dir> [<deck> <deck dir>]
perf_case() {
  local name=$1 aot_exe=$2 aot_dir=$3 jit_exe=$4 jit_dir=$5 deck=${6:-} deck_dir=${7:-}
  local work=$OPSJIT_WORK/perf-run-gpu/$name
  mkdir -p "$work/aot" "$work/jit"
  cp "$aot_dir"/*.in "$work/aot/" 2>/dev/null || true
  cp "$jit_dir"/*.in "$work/jit/" 2>/dev/null || true
  if [ -n "$deck" ]; then cp "$deck_dir/$deck" "$work/aot/clover.in"; cp "$deck_dir/$deck" "$work/jit/clover.in"; fi
  local blk=()
  [ -f "$aot_exe" ] && timed "$name/aot" "$work/aot" "$aot_exe" OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 OPS_BLOCK_SIZE_Z=1
  if [ -f "$jit_exe" ]; then
    (cd "$work/jit" && "$jit_exe" "${blk[@]}" >warm.out 2>&1)
    OPS_JIT_FUSION=1 timed "$name/jit" "$work/jit" "$jit_exe"
    OPS_JIT_FUSION=0 timed "$name/jit-nofuse" "$work/jit" "$jit_exe"
  fi
}

C=$OPS_SRC/apps/c
perf_case clover2d-bm16short "$AOT/apps/c/CloverLeaf/clover_leaf_cuda" "$AOT/apps/c/CloverLeaf/tmp" \
          "$JIT/clover_leaf_jit_cuda" "$JIT/clover_leaf_tmp" clover_bm16_short.in "$C/CloverLeaf"
perf_case clover3d-bm1sshort "$AOT/apps/c/CloverLeaf_3D/clover_leaf_3D_cuda" "$AOT/apps/c/CloverLeaf_3D/tmp" \
          "$JIT/clover_leaf_3D_jit_cuda" "$JIT/clover_leaf_3D_tmp" clover_bm1s_short.in "$C/CloverLeaf_3D"
for p in DP SP SPDP; do
  perf_case tgv-$p "$TGV/tgv_${p}_aot_cuda" "$TGV/tgv_${p}_aot_tmp" "$TGV/tgv_${p}_jit_cuda" "$TGV/tgv_${p}_tmp"
done
echo "results: $OUT"
