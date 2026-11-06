#!/usr/bin/env bash
# Regenerates test/lit/Frontend/Inputs: verbatim copies of real OPS and TGV kernels (see Inputs/README.md for where
# they come from). Run it again only when the sources change; the copies are committed so that the tests do not
# depend on the checkout of OPS.
#
#   OPS_SRC=~/ops-mlir-work/OPS TGV_ROOT=~/ops-mlir-work/reguly-tests/apps/TGV_mixed scripts/make_importer_inputs.sh
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$here/test/lit/Frontend/Inputs
ops=${OPS_SRC:?set OPS_SRC}
tgv=${TGV_ROOT:?set TGV_ROOT}
mkdir -p "$out"

copy() { cp "$ops/apps/c/$1" "$out/$2"; }
copy CloverLeaf/ideal_gas_kernel.h cloverleaf_ideal_gas_kernel.h
copy CloverLeaf/field_summary_kernel.h cloverleaf_field_summary_kernel.h
copy CloverLeaf/calc_dt_kernel.h cloverleaf_calc_dt_kernel.h
copy CloverLeaf/advec_cell_kernel.h cloverleaf_advec_cell_kernel.h
copy CloverLeaf/advec_mom_kernel.h cloverleaf_advec_mom_kernel.h
copy CloverLeaf/generate_chunk_kernel.h cloverleaf_generate_chunk_kernel.h
copy CloverLeaf_3D/PdV_kernel.h cloverleaf3d_PdV_kernel.h
copy TeaLeaf/set_field_kernel.h tealeaf_set_field_kernel.h
copy laplace2d_tutorial/step7/laplace_kernels.h laplace2d_kernels.h
copy multiDim/multidim_kernel.h multidim_kernel.h

# Functions of the TGV headers, verbatim: from the line that starts them to the closing brace in column 0.
extract() { # file function...
  local file=$1
  shift
  for f in "$@"; do
    awk -v name="$f" '
      !on && $0 ~ ("void " name "\\(") { on = 1 }
      on { print }
      on && /^}/ { on = 0; print "" }
    ' "$file"
  done
}
{
  echo "// Verbatim from $(basename "$tgv")/TGsym_DP/opensbliblock00_kernels.h (reguly/tests, apps/TGV_mixed)."
  extract "$tgv/TGsym_DP/opensbliblock00_kernels.h" opensbliblock00Kernel087 opensbliblock00Kernel081 opensbliblock00Kernel002
} >"$out/tgv_dp_kernels.h"
{
  echo "// Verbatim from $(basename "$tgv")/TGsym_SPDP/opensbliblock00_kernels.h (reguly/tests, apps/TGV_mixed)."
  extract "$tgv/TGsym_SPDP/opensbliblock00_kernels.h" opensbliblock00Kernel003 opensbliblock00Kernel005
} >"$out/tgv_spdp_kernels.h"
wc -l "$out"/*.h
