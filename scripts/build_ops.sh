#!/usr/bin/env bash
# Configure, build and install OPS (C/C++ libraries, headers, Python translator) with the pinned
# clang. This is the single source of truth for how we build OPS, locally and on the cluster:
#
#     scripts/build_ops.sh                      # libraries only
#     OPS_BUILD_APPS=ON scripts/build_ops.sh    # also OPS's sample apps and their ctest tests
#
# Environment (defaults come from scripts/env.sh):
#   OPS_SRC            OPS source tree (branch feature/jit-executor)
#   OPS_INSTALL        install prefix (also where the translator venv is created)
#   OPS_CUDA           auto|on|off   (auto: on for the cluster, off locally)
#   GPU_ARCH           CUDA architectures, default "70;80" on the cluster (V100, A100), 89 locally
#   OPS_BUILD_SUFFIX   build-tree suffix, to keep several configurations side by side
#   JOBS               parallel build jobs
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"

BUILD=${OPS_BUILD:-$OPSJIT_BUILD_ROOT/OPS${OPS_BUILD_SUFFIX:-}}
APPS=${OPS_BUILD_APPS:-OFF}
JOBS=${JOBS:-$(nproc)}
OPS_CUDA=${OPS_CUDA:-auto}
[ "$OPS_CUDA" = auto ] && { [ "$OPSJIT_HOST" = cluster ] && OPS_CUDA=on || OPS_CUDA=off; }

args=(
  -S "$OPS_SRC" -B "$BUILD" -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
  -DCMAKE_INSTALL_PREFIX="$OPS_INSTALL"
  -DBUILD_OPS_APPS="$APPS" -DOPS_TEST="$APPS"
  -DAPP_INSTALL_DIR="${OPS_APP_INSTALL:-$OPSJIT_WORK/install/apps}"
)

if [ "$OPS_CUDA" = on ]; then
  GPU_ARCH=${GPU_ARCH:-$([ "$OPSJIT_HOST" = cluster ] && echo "70;80" || echo 89)}
  # nvcc rejects the very new clang as a host compiler: use the system g++ for .cu files only.
  # -no-pie: nvcc's host objects are not position independent, the clang driver links PIE by default.
  args+=( -DGPU_ARCH="$GPU_ARCH" -DGPU_NUMBER=1 -DCMAKE_CUDA_HOST_COMPILER=${OPS_CUDA_HOST_CXX:-/usr/bin/g++}
          -DCUDAToolkit_ROOT="$CUDA_INSTALL_PATH" )
  ld_flags="-no-pie"
else
  args+=( -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON )
fi
# The CUDA language support adds -L/usr/lib/gcc/.../8 (the host compiler nvcc uses) to every link line. The old
# libstdc++ found there lacks symbols that code compiled against the newer headers of the pinned clang needs
# (std::stringstream constructors...). Put the libstdc++ of the clang toolchain first, and run with it.
ld_flags=${ld_flags:-}
stdcxx=$("$CXX" -print-file-name=libstdc++.so 2>/dev/null || true)
if [ "$OPSJIT_HOST" = cluster ] && [ -f "$stdcxx" ]; then
  ld_flags="$ld_flags -L$(dirname "$stdcxx") -Wl,-rpath,$(dirname "$stdcxx")"
fi
[ -n "$ld_flags" ] && args+=( -DCMAKE_EXE_LINKER_FLAGS="$ld_flags" )
# Reproducibility: only look at what env.sh points at.
args+=( -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
        -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE )
# CMake's FindHDF5 asks the h5pcc wrapper where HDF5 is. The shared cluster copy was relocated and its
# wrapper still names the original location (prefix=/home/reguly/projects/tests/hdf5), which holds an
# Intel-built HDF5 that cannot be linked without Intel's runtime libraries. Use a copy of the wrapper
# with the right prefix.
if [ -n "${HDF5_INSTALL_PATH:-}" ] && [ -f "$HDF5_INSTALL_PATH/bin/h5pcc" ]; then
  wrappers=$OPSJIT_BUILD_ROOT/hdf5-wrappers
  mkdir -p "$wrappers"
  sed "s#^prefix=.*#prefix=\"$HDF5_INSTALL_PATH\"#" "$HDF5_INSTALL_PATH/bin/h5pcc" >"$wrappers/h5pcc"
  chmod +x "$wrappers/h5pcc"
  args+=( -DHDF5_ROOT="$HDF5_INSTALL_PATH" -DHDF5_C_COMPILER_EXECUTABLE="$wrappers/h5pcc" )
fi
[ -n "${CFLAG:-}" ] && args+=( -DCFLAG="$CFLAG" )
[ -n "${CXXFLAG:-}" ] && args+=( -DCXXFLAG="$CXXFLAG" )

echo "== configure OPS ($OPS_SRC @ $(git -C "$OPS_SRC" log -1 --format=%h)) -> $BUILD, install $OPS_INSTALL, cuda=$OPS_CUDA apps=$APPS"
cmake "${args[@]}" "$@"
if [ "${OPS_CONFIGURE_ONLY:-0}" = 1 ]; then echo "== configure only"; exit 0; fi
echo "== build"
# -k 0: report every failing target, not only the first
cmake --build "$BUILD" -j "$JOBS" -- -k 0
if [ "$APPS" = ON ] && [ "${OPS_INSTALL_APPS:-0}" != 1 ]; then
  # The apps are tested from the build tree. Installing would also copy every app and input deck.
  echo "== apps built in $BUILD (not installed); run: ctest --test-dir $BUILD"
else
  echo "== install"
  cmake --install "$BUILD"
  echo "OPS installed in $OPS_INSTALL"
fi
