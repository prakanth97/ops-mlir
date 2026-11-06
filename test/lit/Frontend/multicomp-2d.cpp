// Multi-component accessors: the component comes first, ACC<T>(d, x, y); the type of the accessor gets as many
// components as the kernel uses. The kernel also has an idx parameter, which becomes ops.index.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel multidim_kernel --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

#include "Inputs/multidim_kernel.h"

// CHECK-LABEL: ops.kernel @multidim_kernel(%arg0: !ops.acc<f64, 2, 2>) {
// CHECK: ops.index 0
// CHECK: ops.write %arg0 [0, 0], %{{.*}} : !ops.acc<f64, 2, 2>
// CHECK: ops.index 1
// CHECK: ops.write %arg0 [0, 0], %{{.*}} {comp = 1 : i64}
// MANIFEST: "ncomp": 2
// MANIFEST: "kind": "idx"
// MANIFEST: "idx_axes_read": [
// MANIFEST: 0,
// MANIFEST: 1
