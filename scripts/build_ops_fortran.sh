#!/usr/bin/env bash
# Builds OPS's Fortran f2c library (libops_f2c_seq.a and the .mod files of its Fortran API) with flang, for the
# Fortran applications of apps/CMakeLists.txt (OPSJIT_APPS_FORTRAN=ON). The outputs go to $OPS_SRC/ops/fortran/{lib,mod}/clang.
#
#     scripts/build_ops_fortran.sh
#
# Environment: as scripts/env.sh (LLVM_INSTALL_PATH must have flang), JOBS.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$here/env.sh"

FC=${FC:-$LLVM_INSTALL_PATH/bin/flang}
[ -x "$FC" ] || { echo "build_ops_fortran.sh: no flang at $FC" >&2; exit 1; }
make -C "$OPS_SRC/ops/fortran" -j "${JOBS:-$(nproc)}" f2c_seq \
     OPS_INSTALL_PATH="$OPS_SRC/ops" OPS_COMPILER=clang CC="$CC" CXX="$CXX" FC="$FC"
# OPS's makefiles give flang no module directory: the .mod files land in ops/fortran. When there is nothing to rebuild
# (a rerun) there are none to move, and the ones moved by the previous run stay where they are.
mkdir -p "$OPS_SRC/ops/fortran/mod/clang"
shopt -s nullglob
mods=( "$OPS_SRC"/ops/fortran/*.mod )
[ ${#mods[@]} -gt 0 ] && mv -f "${mods[@]}" "$OPS_SRC/ops/fortran/mod/clang/"
[ -f "$OPS_SRC/ops/fortran/mod/clang/ops_fortran_declarations.mod" ] || { echo "build_ops_fortran.sh: no ops_fortran_declarations.mod after the build" >&2; exit 1; }
echo "== $OPS_SRC/ops/fortran/lib/clang/libops_f2c_seq.a, $OPS_SRC/ops/fortran/mod/clang"
