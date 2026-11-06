// Calls of functions the application defines are inlined: scalar by-value parameters, a body that ends in
// `return value;`. Other helpers are rejected with a reason.
// REQUIRES: ops-headers
// RUN: ops-kernel-import --kernel uses_helper --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include | FileCheck %s
// RUN: ops-kernel-import --kernel uses_loop_helper --source %s --ndim 2 -o %t.mlir --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: FileCheck %s --check-prefix=REJECT-MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

static inline double scaled(double a, double b) {
  double t = a * b;
  return t + 1.0;
}

static inline double twice(double a) { return scaled(a, 2.0); }

static inline double early(double a) {
  if (a > 0.0) return a;
  return -a;
}

void uses_helper(const ACC<double> &v, ACC<double> &u) { u(0, 0) = twice(v(0, 0)) - scaled(v(1, 0), 3.0); }

void uses_loop_helper(const ACC<double> &v, ACC<double> &u) { u(0, 0) = early(v(0, 0)); }

// CHECK-LABEL: ops.kernel @uses_helper
// CHECK: %[[A:.*]] = ops.read %arg0 [0, 0]
// CHECK: arith.mulf %[[A]], %{{.*}}
// CHECK: arith.addf
// CHECK: ops.read %arg0 [1, 0]
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: ops.write %arg1 [0, 0]

// REJECT: more than one return
// REJECT-MANIFEST: "status": "failed"
