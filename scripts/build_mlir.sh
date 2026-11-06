#!/usr/bin/env bash
# Configure and build ops-mlir (tools and libopsjit) without running the tests (scripts/ci.sh does that).
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
