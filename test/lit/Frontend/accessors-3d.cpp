// The fourth-order derivative of TGV with mixed precision (reads double, writes float): float constants are computed
// in float and converted, as in C.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel opensbliblock00Kernel005 --source %s --ndim 3 -o - --manifest %t.d.json -- -I%ops_include | FileCheck %s --check-prefix=SPDP
// RUN: FileCheck %s --check-prefix=SPDP-MANIFEST --input-file=%t.d.json

#include <ops_lib_core.h>

// The application's globals (TGV's constants.h).
extern double invDelta0block0;

#include "Inputs/tgv_spdp_kernels.h"

// SPDP-LABEL: ops.kernel @opensbliblock00Kernel005(%arg0: !ops.acc<f64, 3, 1>, %arg1: !ops.acc<f32, 3, 1>)
// SPDP: arith.constant 2.000000e+00 : f32
// SPDP: arith.divf {{.*}} : f32
// SPDP: arith.extf {{.*}} : f32 to f64
// SPDP: ops.read %arg0 [-1, 0, 0]
// SPDP: ops.read %arg0 [2, 0, 0]
// SPDP: ops.read %arg0 [-2, 0, 0]
// SPDP: ops.read %arg0 [1, 0, 0]
// SPDP: ops.const_ref @invDelta0block0 : f64
// SPDP: arith.truncf {{.*}} : f64 to f32
// SPDP: ops.write %arg1 [0, 0, 0], {{.*}} : !ops.acc<f32, 3, 1>
// SPDP-MANIFEST: "consts": [
// SPDP-MANIFEST: "name": "invDelta0block0",
// SPDP-MANIFEST: "elem": "f64",
// SPDP-MANIFEST: "kind": "acc",
// SPDP-MANIFEST: "elem": "f32",
// SPDP-MANIFEST: "kind": "acc",
