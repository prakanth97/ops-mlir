// Offsets that depend on data: the donor cell of the advection scheme of CloverLeaf is chosen by the sign of the flux.
// Reads become ops.read_dyn; the registered struct constant `field` is read member by member (byte offsets from the
// record layout).
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel advec_cell_kernel3_xdir --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include -I%S/Inputs | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

#include "Inputs/cloverleaf_advec_cell_kernel.h"

// CHECK-LABEL: ops.kernel @advec_cell_kernel3_xdir(
// field.x_max is the second int of the struct
// CHECK: ops.const_ref @field {offset = 4 : i64} : i32
// The four indices that pick the donor cell leave the branches as results of nested scf.ifs.
// CHECK: scf.if %{{.*}} -> (i32, i32, i32, i32) {
// CHECK: } else {
// CHECK: scf.if %{{.*}} -> (i32, i32, i32, i32) {
// Reads at the offsets they chose are data-dependent.
// CHECK: ops.read_dyn %arg1[%{{.*}}, %{{.*}} : i32, i32]
// CHECK: ops.read_dyn %arg3[%{{.*}}, %{{.*}} : i32, i32]
// CHECK: ops.write %arg6 [0, 0]
// CHECK: ops.write %arg7 [0, 0]
// MANIFEST: "consts": [
// MANIFEST: "name": "field",
// MANIFEST: "offset": 4
// MANIFEST: "dynamic_reads": [
// MANIFEST: "1",
