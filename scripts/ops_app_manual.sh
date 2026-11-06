#!/usr/bin/env bash
# Translate and build ONE sequential/OpenMP OPS app by hand, outside OPS's CMake app machinery
# (whose configure step translates every sample app and takes minutes). For quick local checks.
#
#   scripts/ops_app_manual.sh <app dir relative to OPS/apps/c> <main source> <out dir> [extra clang flags...]
#   scripts/ops_app_manual.sh laplace2d_tutorial/step7 laplace2d.cpp /tmp/lap
#
# Produces <out>/app_eager (stock OPS, no lazy queue in the stubs) and <out>/app_lazy (stubs compiled
# with -DOPS_LAZY, i.e. descriptors; run with OPS_TILING or OPS_LAZY_QUEUE to queue). Needs the
# installed OPS (scripts/build_ops.sh) including its translator venv.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"

app=$1; main=$2; out=$3; shift 3
SRC=$OPS_SRC/apps/c/$app
TR=$OPS_INSTALL/translator/ops_translator
mkdir -p "$out/tmp"
cp "$SRC"/*.cpp "$SRC"/*.h "$out/tmp/" 2>/dev/null || true
cp "$SRC"/*.in "$out/tmp/" 2>/dev/null || true
cd "$out/tmp"
"$TR/ops_venv/bin/python3" "$TR/ops-translator" -DOPS_ACC_IGNORE -I "$OPS_INSTALL/include" --file_paths "$main" >/dev/null

gen=$(ls ./*ops*.cpp | head -1)
common=(-O2 -fopenmp -I"$OPS_INSTALL/include" -I. "$@")
libs=("$OPS_INSTALL/lib/libops_seq.a" -Wl,-rpath,"$OPSJIT_OMP_LIB_DIR")
# app_eager: the translated program without OPS_LAZY
"$CXX" "${common[@]}" "$gen" mpi_openmp/mpi_openmp_kernels.cpp "${libs[@]}" -o "$out/app_eager"
# app_lazy: loops are enqueued as descriptors
"$CXX" "${common[@]}" -DOPS_LAZY "$gen" mpi_openmp/mpi_openmp_kernels.cpp "${libs[@]}" -o "$out/app_lazy"
echo "built $out/app_eager and $out/app_lazy"
