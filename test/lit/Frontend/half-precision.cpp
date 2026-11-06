// Half precision (_Float16, OPS's `half` on hosts that have it): accessors and locals of f16, mixed with float.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel mixed --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>
#include <math.h>

void mixed(ACC<_Float16> &a, const ACC<float> &b, const ACC<_Float16> &c) {
  float t = b(0, 0) * cosf(b(0, 0));
  a(0, 0) = t + c(0, 0);
}

// CHECK-LABEL: ops.kernel @mixed(%arg0: !ops.acc<f16, 2, 1>, %arg1: !ops.acc<f32, 2, 1>, %arg2: !ops.acc<f16, 2, 1>)
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK: ops.write %arg0 [0, 0]
// MANIFEST: "elem": "f16"
