#!/usr/bin/env bash
# Environment for building and testing ops-mlir v2 and OPS. SOURCE it (do not execute):
#
#     source scripts/env.sh
#
# It detects where it runs:
#   * the `cl` cluster (login or compute node)  -> shared LLVM install, gcc-toolset-13, Lmod modules
#   * local WSL                                  -> rsync copy of the same LLVM install
# and exports the same variable names on both, so build and test scripts are identical.
# Override any variable before sourcing.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "scripts/env.sh must be sourced from bash" >&2
  return 1 2>/dev/null || exit 1
fi

_opsjit_host_kind() {
  if [ -d "$HOME/MLIR/lib_install" ] && [ -d /home/shared/software ]; then echo cluster; else echo local; fi
}
export OPSJIT_HOST=${OPSJIT_HOST:-$(_opsjit_host_kind)}

if [ "$OPSJIT_HOST" = cluster ]; then
  export OPSJIT_WORK=${OPSJIT_WORK:-$HOME/ops-jit-cl}
  export LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-$HOME/MLIR/lib_install/llvm-project-23.1.0-CUDA-Release}
  # gcc-toolset-13 supplies the libstdc++ the clang 23 binaries were built against.
  if [ -f /opt/rh/gcc-toolset-13/enable ]; then
    # shellcheck disable=SC1091
    source /opt/rh/gcc-toolset-13/enable
  fi
  # Lmod: OP2-env gives OpenMPI 4.1.6, CUDA 12.3, HDF5 (parallel and serial) paths.
  if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/lmod.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.6 >/dev/null 2>&1 || true
    module load OP2-env >/dev/null 2>&1 || module load gpu/cuda/12.3 >/dev/null 2>&1 || true
  fi
  export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/home/shared/software/cuda/12.3}
  export MPI_INSTALL_PATH=${MPI_INSTALL_PATH:-/home/shared/software/openmpi/4.1.6-gcc8.5}
  export HDF5_INSTALL_PATH=${HDF5_INSTALL_PATH:-/home/shared/software/op-dsl/hdf5}
  export HDF5_SEQ_INSTALL_PATH=${HDF5_SEQ_INSTALL_PATH:-/home/shared/software/op-dsl/hdf5-seq}
else
  export OPSJIT_WORK=${OPSJIT_WORK:-$HOME/ops-mlir-work}
  export LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-$HOME/ops-mlir-work/toolchain/llvm-23.1.0}
  export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/usr/local/cuda}
  [ -d "$CUDA_INSTALL_PATH" ] || CUDA_INSTALL_PATH=/usr
  export MPI_INSTALL_PATH=${MPI_INSTALL_PATH:-}      # optional locally; MPI runs are tested on the cluster
  export HDF5_INSTALL_PATH=${HDF5_INSTALL_PATH:-}
  export HDF5_SEQ_INSTALL_PATH=${HDF5_SEQ_INSTALL_PATH:-}
  # LLVM's CMake exports require the ZLIB::ZLIB target. This WSL image has the zlib runtime but no
  # development files (zlib.h, libz.so) and no sudo: hand CMake the runtime library directly (no header
  # is needed, nothing here includes zlib.h).
  if [ ! -f /usr/include/zlib.h ] && [ -e /usr/lib/x86_64-linux-gnu/libz.so.1 ]; then
    OPSJIT_CMAKE_ARGS="${OPSJIT_CMAKE_ARGS:-} -DZLIB_INCLUDE_DIR=/usr/include -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so.1"
  fi
  # Same for zstd (see the OPSJIT_ZSTD_LIBRARY shim in the top-level CMakeLists.txt).
  if [ ! -f /usr/include/zstd.h ] && [ -e /usr/lib/x86_64-linux-gnu/libzstd.so.1 ]; then
    OPSJIT_CMAKE_ARGS="${OPSJIT_CMAKE_ARGS:-} -DOPSJIT_ZSTD_LIBRARY=/usr/lib/x86_64-linux-gnu/libzstd.so.1"
  fi
fi
# Extra CMake arguments needed on this host when configuring against the pinned LLVM (word-split on use).
export OPSJIT_CMAKE_ARGS=${OPSJIT_CMAKE_ARGS:-}

if [ ! -x "$LLVM_INSTALL_PATH/bin/clang" ]; then
  echo "scripts/env.sh: no clang in LLVM_INSTALL_PATH=$LLVM_INSTALL_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

# One install is LLVM, MLIR, clang and the OpenMP runtime.
export LLVM_DIR=$LLVM_INSTALL_PATH/lib/cmake/llvm
export MLIR_DIR=$LLVM_INSTALL_PATH/lib/cmake/mlir
export Clang_DIR=$LLVM_INSTALL_PATH/lib/cmake/clang
export CC=$LLVM_INSTALL_PATH/bin/clang
export CXX=$LLVM_INSTALL_PATH/bin/clang++
export OPSJIT_OMP_LIB_DIR=$LLVM_INSTALL_PATH/lib/x86_64-unknown-linux-gnu   # libomp.so (LLVM OpenMP runtime)
export OPSJIT_CUDA_RUNTIME=$LLVM_INSTALL_PATH/lib/libmlir_cuda_runtime.so
export OPS_COMPILER=${OPS_COMPILER:-clang}

# Host compiler nvcc uses (the cluster has /usr/bin/g++; this WSL image only g++-12).
if [ -z "${OPS_CUDA_HOST_CXX:-}" ] && [ ! -x /usr/bin/g++ ]; then
  export OPS_CUDA_HOST_CXX=$(ls /usr/bin/g++-* 2>/dev/null | sort -V | tail -1)
fi

# Source trees. Locally they live directly under ~/ops-mlir-work; on the cluster the work trees are
# updated by cluster/v2/sync.sh and live under ~/ops-jit-cl/work.
# The OPS source tree defaults to the submodule third_party/OPS of this repository when it is checked out
# (git submodule update --init third_party/OPS); OPS_SRC overrides it (a developer working on the OPS side points it at
# their own clone).
_opsjit_sub=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/OPS
if [ "$OPSJIT_HOST" = cluster ]; then
  _opsjit_ops=$OPSJIT_WORK/work/OPS
  export OPSMLIR_SRC=${OPSMLIR_SRC:-$OPSJIT_WORK/work/ops-mlir}
else
  _opsjit_ops=$OPSJIT_WORK/OPS
  export OPSMLIR_SRC=${OPSMLIR_SRC:-$OPSJIT_WORK/ops-mlir}
fi
[ -f "$_opsjit_sub/CMakeLists.txt" ] && _opsjit_ops=$_opsjit_sub
export OPS_SRC=${OPS_SRC:-$_opsjit_ops}
# TGV_mixed benchmark apps (github.com/reguly/tests, apps/TGV_mixed).
export TGV_ROOT=${TGV_ROOT:-$OPSJIT_WORK/reguly-tests/apps/TGV_mixed}
# On the cluster the benchmark repository is already checked out in the project area (read only use).
[ -d "$TGV_ROOT" ] || { [ -d "$HOME/projects/tests/apps/TGV_mixed" ] && export TGV_ROOT=$HOME/projects/tests/apps/TGV_mixed; }

# Install prefix for OPS libraries/headers built by scripts/build_ops.sh; build trees go under build/.
export OPS_INSTALL=${OPS_INSTALL:-$OPSJIT_WORK/install/ops}
export OPSJIT_BUILD_ROOT=${OPSJIT_BUILD_ROOT:-$OPSJIT_WORK/build}

export PATH=$LLVM_INSTALL_PATH/bin:$CUDA_INSTALL_PATH/bin:${MPI_INSTALL_PATH:+$MPI_INSTALL_PATH/bin:}$PATH
export LD_LIBRARY_PATH=$LLVM_INSTALL_PATH/lib:$OPSJIT_OMP_LIB_DIR:$CUDA_INSTALL_PATH/lib64:${MPI_INSTALL_PATH:+$MPI_INSTALL_PATH/lib:}${HDF5_INSTALL_PATH:+$HDF5_INSTALL_PATH/lib:}${LD_LIBRARY_PATH:-}

# Pin threads (the CPU backends are bandwidth-bound; unpinned runs are noisy).
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

# lit (not vendored): the pinned LLVM install ships FileCheck/not/count but no llvm-lit; use the python lit package
# (pip install -r requirements-test.txt) or the one of an llvm-project build via OPSJIT_LIT.
export OPSJIT_LIT=${OPSJIT_LIT:-$(command -v lit || command -v llvm-lit || true)}

echo "ops-mlir env: host=$OPSJIT_HOST llvm=$("$LLVM_INSTALL_PATH/bin/llvm-config" --version) clang=$(basename "$CXX") work=$OPSJIT_WORK" >&2
