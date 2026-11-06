// ops-outline-groups: every loop of a chain becomes its own chain, the original chain calls them.
// RUN: ops-opt %s -ops-fuse-point-local -ops-forward-stores -ops-outline-groups | FileCheck %s
// RUN: ops-opt %s -ops-fuse-point-local -ops-forward-stores -ops-outline-groups | ops-opt | FileCheck %s

// Two point-wise loops fuse into one group; a reduction over the result is a barrier and a group of its own; the
// loop after it stays behind it. The constant the first loop captured from the chain is copied into its group.
// CHECK-LABEL: ops.chain @group_0
// CHECK-SAME: (%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: !ops.dat<f64, 2, 1>, %arg5: !ops.dat<f64, 2, 1>, %arg6: !ops.dat<f64, 2, 1>)
// CHECK-SAME: attributes {ops.range_values = [array<i64: 0, 8, 0, 8>]}
// CHECK: arith.constant 2.000000e+00 : f64
// CHECK: ops.par_loop
// CHECK-SAME: ops.fused_loops = [1, 2]
// CHECK: ops.return
//
// CHECK-LABEL: ops.chain @group_1
// CHECK-SAME: -> f64
// CHECK-SAME: ops.range_values = [array<i64: 0, 8, 0, 8>]
// CHECK: %[[R:.*]] = ops.par_loop
// CHECK-SAME: reduction_kinds = [#ops.red_kind<min>]
// CHECK: ops.return %[[R]] : f64
//
// CHECK-LABEL: ops.chain @group_2
// CHECK: ops.par_loop
// CHECK: ops.return
//
// CHECK-LABEL: ops.chain @chain
// CHECK-NOT: ops.par_loop
// CHECK: ops.call @group_0(%arg3, %arg4, %arg5, %arg6, %arg0, %arg1, %arg2) : (
// CHECK: %[[M:.*]] = ops.call @group_1(
// CHECK: ops.call @group_2(
// CHECK: ops.return %[[M]] : f64
ops.chain @chain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %c: !ops.dat<f64, 2, 1>, %lo: index, %hi: index,
                 %lo1: index, %hi1: index) -> f64
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>, array<i64: 0, 8, 0, 8>,
                                      array<i64: 0, 8, 0, 8>]} {
  %two = arith.constant 2.0 : f64
  ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() attributes {loop_id = 1 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %w = arith.mulf %v, %two : f64
    ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  }
  ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() attributes {loop_id = 2 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %v : !ops.acc<f64, 2, 1>
  }
  %m = ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%c : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<min>], loop_id = 3 : i64} -> f64 {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %r: !ops.red<f64>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.reduce %r, %v : !ops.red<f64>
  }
  ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() attributes {loop_id = 4 : i64} {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
    %z = arith.constant 0.0 : f64
    ops.write %x[0, 0], %z : !ops.acc<f64, 2, 1>
  }
  ops.return %m : f64
}
