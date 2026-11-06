// Operations of kernel bodies.
// RUN: ops-opt %s --verify-roundtrip | FileCheck %s
// RUN: ops-opt %s | ops-opt | FileCheck %s

// CHECK-LABEL: ops.kernel @everything
ops.kernel @everything(%a: !ops.acc<f64, 2, 3>, %b: !ops.acc<f64, 2, 1>, %g: !ops.gbl<f64, 2>, %r: !ops.red<f64>,
                       %n: !ops.acc<i32, 2, 1>) {
  // CHECK: ops.read %{{.*}} [1, -1] : !ops.acc<f64, 2, 3> -> f64
  %v0 = ops.read %a[1, -1] : !ops.acc<f64, 2, 3> -> f64
  // CHECK: ops.read %{{.*}} [0, 0] {comp = 2 : i64} : !ops.acc<f64, 2, 3> -> f64
  %v1 = ops.read %a[0, 0] {comp = 2 : i64} : !ops.acc<f64, 2, 3> -> f64
  %dx = ops.index 0
  %dy = ops.index 1
  // CHECK: ops.read_dyn %{{.*}}[%{{.*}}, %{{.*}} : i32, i32] : !ops.acc<f64, 2, 3> -> f64
  %v2 = ops.read_dyn %a[%dx, %dy : i32, i32] : !ops.acc<f64, 2, 3> -> f64
  // CHECK: ops.gbl_read %{{.*}}[1] : !ops.gbl<f64, 2> -> f64
  %g1 = ops.gbl_read %g[1] : !ops.gbl<f64, 2> -> f64
  // CHECK: ops.const_ref @dt : f64
  %dt = ops.const_ref @dt : f64
  // CHECK: ops.const_ref @field {offset = 8 : i64} : i32
  %xmax = ops.const_ref @field {offset = 8 : i64} : i32
  // CHECK: ops.const_ref @states[%{{.*}} : i32] {array_stride = 72 : i64, offset = 8 : i64} : f64
  %e = ops.const_ref @states[%dx : i32] {offset = 8 : i64, array_stride = 72 : i64} : f64
  %s0 = arith.addf %v0, %v1 : f64
  %s1 = arith.addf %s0, %v2 : f64
  %s2 = arith.addf %s1, %g1 : f64
  %s3 = arith.mulf %s2, %dt : f64
  %s4 = arith.addf %s3, %e : f64
  // CHECK: ops.write %{{.*}} [-1, 0], %{{.*}} : !ops.acc<f64, 2, 1>
  ops.write %b[-1, 0], %s4 : !ops.acc<f64, 2, 1>
  // CHECK: ops.reduce %{{.*}}, %{{.*}} : !ops.red<f64>
  ops.reduce %r, %s4 : !ops.red<f64>
  ops.return
}
