#!/usr/bin/env bash
# Configure, build and run the CPU test suites. Runs locally (WSL) and on a cluster node; the cluster
# counterpart that adds the MPI and GPU suites is `ctest -L mpi` / `ctest -L gpu` in a SLURM job.
#
#   scripts/ci.sh            configure (if needed), build, check-ops-mlir, ctest -L cpu
#   BUILD=/path scripts/ci.sh
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"
src=$(cd "$here/.." && pwd)
BUILD=${BUILD:-$OPSJIT_BUILD_ROOT/ops-mlir}

cmake -S "$src" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" \
  -DOPS_INSTALL="$OPS_INSTALL" -DOPSJIT_OPS_SRC="$OPS_SRC" $OPSJIT_CMAKE_ARGS "$@"
ninja -C "$BUILD"
ninja -C "$BUILD" check-ops-mlir
# End-to-end suites (cpu label) are added from P3.9 on.
if ctest --test-dir "$BUILD" -N -L cpu 2>/dev/null | grep -q "Total Tests: [1-9]"; then
  ctest --test-dir "$BUILD" -L cpu --output-on-failure
fi
echo "ci.sh: OK"
