// The planning half of the JIT, as the executor runs it on a queue: kernels are materialised, bound constants folded,
// loops fused and every group outlined.
// RUN: ops-opt %s -ops-jit-frontend -ops-jit-plan | FileCheck %s
// RUN: ops-opt %s -pass-pipeline="builtin.module(ops-jit-frontend,ops-jit-plan{max-loops=1})" | FileCheck %s --check-prefix=NOFUSE

ops.kernel @scale(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>) {
  %k = ops.const_ref @factor : f64
  %v = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
  %w = arith.mulf %v, %k : f64
  ops.write %y[0], %w : !ops.acc<f64, 1, 1>
  ops.return
}
ops.kernel @shift(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>) {
  %v = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
  ops.write %y[0], %v : !ops.acc<f64, 1, 1>
  ops.return
}

// factor = 3.0 is bound: it is folded into the first loop. The two loops are point-wise and fuse into one group.
// CHECK-LABEL: ops.chain @group_0
// CHECK: arith.constant 3.000000e+00 : f64
// CHECK: ops.par_loop
// CHECK-SAME: ops.fused_loops
// CHECK-LABEL: ops.chain @chain
// CHECK-NOT: ops.par_loop
// CHECK: ops.call @group_0
// CHECK-NOT: ops.call
// NOFUSE: ops.chain @group_0
// NOFUSE: ops.chain @group_1
// NOFUSE: ops.chain @chain
// NOFUSE: ops.call @group_0
// NOFUSE: ops.call @group_1
ops.chain @chain(%a: !ops.dat<f64, 1, 1>, %b: !ops.dat<f64, 1, 1>, %c: !ops.dat<f64, 1, 1>, %lo0: index, %hi0: index,
                 %lo1: index, %hi1: index)
    attributes {ops.const_roles = {factor = "bound"}, ops.const_values = {factor = array<i8: 0, 0, 0, 0, 0, 0, 8, 64>},
                "ops.range_values" = [array<i64: 0, 16>, array<i64: 0, 16>]} {
  ops.par_loop @scale range(%lo0, %hi0)
      dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>, %b : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>):
  }
  ops.par_loop @shift range(%lo1, %hi1)
      dats(%b : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>, %c : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>):
  }
  ops.return
}
