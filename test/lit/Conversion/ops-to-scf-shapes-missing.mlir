// ops-specialize mode=shapes needs ops.shape_values.
// RUN: not ops-opt %s -ops-specialize=mode=shapes 2>&1 | FileCheck %s

// CHECK: ops-specialize mode=shapes needs an `ops.shape_values` entry
ops.chain @unshaped(%lo0: index, %hi0: index, %a: !ops.dat<f64, 1, 1>) attributes {ops.range_values = [array<i64: 0, 4>]} {
  ops.par_loop range(%lo0, %hi0) dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>):
  }
  ops.return
}
