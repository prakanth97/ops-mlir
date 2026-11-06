#!/usr/bin/env bash
# Configure and build the OPS applications through the JIT toolchain (apps/CMakeLists.txt): the *_jit* executables.
#
#     scripts/build_apps.sh                 # needs: scripts/build_ops.sh (OPS install), ops-mlir built (ninja opsjit ops-kernel-import)
#
# Environment: as scripts/env.sh, plus APPS_CMAKE_ARGS (extra cmake arguments, e.g. -DOPSJIT_APPS=tgv_DP;tgv_SP), APPS_BUILD (default $OPSJIT_BUILD_ROOT/OPS-jit-apps), OPSJIT_BUILD, JOBS, OPSJIT_APPS_CUDA=ON|OFF
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"

BUILD=${APPS_BUILD:-$OPSJIT_BUILD_ROOT/OPS-jit-apps}
OPSJIT_BUILD=${OPSJIT_BUILD:-$OPSJIT_BUILD_ROOT/ops-mlir}
JOBS=${JOBS:-$(nproc)}
CUDA=${OPSJIT_APPS_CUDA:-$([ "$OPSJIT_HOST" = cluster ] && echo ON || echo OFF)}

args=( -S "$here/../apps" -B "$BUILD" -G Ninja
       -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
       -DOPS_INSTALL="$OPS_INSTALL" -DOPS_SRC="$OPS_SRC" -DOPSJIT_BUILD="$OPSJIT_BUILD" -DTGV_ROOT="$TGV_ROOT"
       -DOPSJIT_APPS_CUDA="$CUDA" -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF )
if [ "$CUDA" = ON ]; then
  GPU_ARCH=${GPU_ARCH:-$([ "$OPSJIT_HOST" = cluster ] && echo "70;80" || echo 89)}
  args+=( -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" -DCMAKE_CUDA_HOST_COMPILER=${OPS_CUDA_HOST_CXX:-/usr/bin/g++}
          -DCUDAToolkit_ROOT="$CUDA_INSTALL_PATH" )
fi
# Same linker setup as scripts/build_ops.sh (libstdc++ of the clang toolchain on the cluster; -no-pie for nvcc objects).
ld_flags=""
[ "$CUDA" = ON ] && ld_flags="-no-pie"
stdcxx=$("$CXX" -print-file-name=libstdc++.so 2>/dev/null || true)
if [ "$OPSJIT_HOST" = cluster ] && [ -f "$stdcxx" ]; then
  ld_flags="$ld_flags -L$(dirname "$stdcxx") -Wl,-rpath,$(dirname "$stdcxx")"
fi
[ -n "$ld_flags" ] && args+=( -DCMAKE_EXE_LINKER_FLAGS="$ld_flags" )
if [ -n "${HDF5_INSTALL_PATH:-}" ]; then
  wrappers=$OPSJIT_BUILD_ROOT/hdf5-wrappers
  args+=( -DHDF5_ROOT="$HDF5_INSTALL_PATH" -DHDF5_C_COMPILER_EXECUTABLE="$wrappers/h5pcc" -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE )
fi
cmake "${args[@]}" ${APPS_CMAKE_ARGS:-} "$@"
cmake --build "$BUILD" -j "$JOBS" -- -k 0
echo "== apps built in $BUILD; run: ctest --test-dir $BUILD -L jit-app"
