// Pointer-style kernels with the OPS_ACCn macros: PdV of CloverLeaf 3D. The parameters are plain pointers; the
// accessor an index belongs to is the n of OPS_ACCn.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel PdV_kernel_predict --source %s --ndim 3 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

// A global of the application (CloverLeaf 3D data.h).
extern double dt;

#include "Inputs/cloverleaf3d_PdV_kernel.h"

// The 14 parameters are inferred: pointer parameters indexed with OPS_ACCn are dats, the others would be globals.
// CHECK-LABEL: ops.kernel @PdV_kernel_predict(%arg0: !ops.acc<f64, 3, 1>, %arg1: !ops.acc<f64, 3, 1>, %arg2: !ops.acc<f64, 3, 1>, %arg3: !ops.acc<f64, 3, 1>, %arg4: !ops.acc<f64, 3, 1>
// OPS_ACC0(0,0,0) is the centre point of the first accessor, OPS_ACC1(0,1,1) the corner of the second.
// CHECK: ops.read %arg0 [0, 0, 0]
// CHECK: ops.read %arg1 [0, 0, 0]
// CHECK: ops.read %arg1 [0, 1, 0]
// CHECK: ops.read %arg1 [0, 0, 1]
// CHECK: ops.read %arg1 [0, 1, 1]
// CHECK: ops.const_ref @dt : f64
// CHECK: ops.write %arg4 [0, 0, 0]
// CHECK: ops.write %arg8 [0, 0, 0]
// MANIFEST: "kind": "acc",
// MANIFEST: "legacy_macro": true,
