// ops-specialize-consts: bound constants folded, the loops that index them unrolled.
// RUN: ops-opt %s -ops-materialize-kernels -ops-specialize-consts | FileCheck %s

// The number of states bounds a loop over a table of states; both are constants of the application.
ops.kernel @eos(%p: !ops.acc<f64, 2, 1>) {
  %n = ops.const_ref @number_of_states : i32
  %lo = arith.constant 0 : index
  %hi = arith.index_cast %n : i32 to index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %sum = scf.for %i = %lo to %hi step %c1 iter_args(%acc = %zero) -> (f64) {
    %ii = arith.index_cast %i : index to i32
    %e = ops.const_ref @states[%ii : i32] {array_stride = 8 : i64} : f64
    %next = arith.addf %acc, %e : f64
    scf.yield %next : f64
  }
  ops.write %p[0, 0], %sum : !ops.acc<f64, 2, 1>
  ops.return
}

ops.kernel @scaled(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  %dt = ops.const_ref @dt : f64
  %k = ops.const_ref @free_parameter : f64
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = arith.mulf %v, %dt : f64
  %z = arith.addf %w, %k : f64
  ops.write %y[0, 0], %z : !ops.acc<f64, 2, 1>
  ops.return
}

// dt = 0.25; number_of_states = 3; states = [1.0, 2.0, 0.5]; free_parameter has no value and stays an argument
// CHECK-LABEL: ops.chain @chain
// CHECK-SAME: %arg6: f64) attributes {ops.binding_consts = [{arg = 6 : i64, name = "free_parameter", offset = 0 : i64}]
// CHECK-DAG: %[[SUM:.*]] = arith.constant 3.500000e+00 : f64
// CHECK-DAG: %[[DTC:.*]] = arith.constant 2.500000e-01 : f64
// CHECK: ops.par_loop @eos
// CHECK-SAME: scalars()
// CHECK-NOT: scf.for
// CHECK: ops.write {{.*}} [0, 0], %[[SUM]]
// CHECK: ops.par_loop @scaled
// CHECK-SAME: scalars(%arg6 : f64)
// CHECK: arith.mulf {{.*}}, %[[DTC]]
// CHECK: arith.addf
// CHECK-NOT: ops.const_ref
ops.chain @chain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index,
                 %hi1: index)
    attributes {ops.const_roles = {dt = "bound", number_of_states = "bound", states = "bound"},
                ops.const_values = {dt = array<i8: 0, 0, 0, 0, 0, 0, -48, 63>,
                                    number_of_states = array<i8: 3, 0, 0, 0>,
                                    states = array<i8: 0, 0, 0, 0, 0, 0, -16, 63, 0, 0, 0, 0, 0, 0, 0, 64,
                                                       0, 0, 0, 0, 0, 0, -32, 63>}} {
  ops.par_loop @eos range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop @scaled range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.return
}
