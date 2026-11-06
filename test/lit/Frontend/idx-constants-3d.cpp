// The initial condition of TGV: the global index (idx), registered constants read from the application, trigonometric
// functions, many outputs; and a kernel that divides two dats.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel opensbliblock00Kernel087 --source %s --ndim 3 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json
// RUN: ops-kernel-import --kernel opensbliblock00Kernel002 --source %s --ndim 3 -o - --manifest %t.json -- -I%ops_include | FileCheck %s --check-prefix=DIV

#include <ops_lib_core.h>

extern double Delta0block0, Delta1block0, Delta2block0, Minf, gama;

#include "Inputs/tgv_dp_kernels.h"

// CHECK-LABEL: ops.kernel @opensbliblock00Kernel087(%arg0: !ops.acc<f64, 3, 1>, %arg1: !ops.acc<f64, 3, 1>, %arg2: !ops.acc<f64, 3, 1>, %arg3: !ops.acc<f64, 3, 1>, %arg4: !ops.acc<f64, 3, 1>)
// CHECK: ops.const_ref @Delta0block0 : f64
// CHECK: ops.index 0
// CHECK: ops.const_ref @Delta1block0 : f64
// CHECK: ops.index 1
// CHECK: ops.const_ref @Delta2block0 : f64
// CHECK: ops.index 2
// CHECK: math.cos
// CHECK: math.sin
// CHECK: ops.const_ref @Minf : f64
// CHECK: ops.const_ref @gama : f64
// CHECK: ops.write %arg1 [0, 0, 0]
// CHECK: ops.write %arg0 [0, 0, 0]
// MANIFEST: "consts": [
// MANIFEST: "name": "Delta0block0"
// MANIFEST: "kind": "idx"
// MANIFEST: "idx_axes_read": [
// MANIFEST: 0,
// MANIFEST: 1,
// MANIFEST: 2

// DIV-LABEL: ops.kernel @opensbliblock00Kernel002(%arg0: !ops.acc<f64, 3, 1>, %arg1: !ops.acc<f64, 3, 1>, %arg2: !ops.acc<f64, 3, 1>)
// DIV: %[[N:.*]] = ops.read %arg1 [0, 0, 0]
// DIV: %[[D:.*]] = ops.read %arg0 [0, 0, 0]
// DIV: %[[Q:.*]] = arith.divf %[[N]], %[[D]]
// DIV: ops.write %arg2 [0, 0, 0], %[[Q]]
