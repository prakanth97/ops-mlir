// ops-materialize-kernels: constants and globals read inside the regions of the kernel (under an `if`) become loop
// operands like the ones read at the top level of the body.
// RUN: ops-opt %s -ops-materialize-kernels | FileCheck %s

ops.kernel @guarded(%x: !ops.acc<f64, 2, 1>, %g: !ops.gbl<f64, 1>, %y: !ops.acc<f64, 2, 1>) {
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %zero = arith.constant 0.0 : f64
  %c = arith.cmpf ogt, %v, %zero : f64
  scf.if %c {
    %dt = ops.const_ref @dt : f64
    %a = ops.gbl_read %g[0] : !ops.gbl<f64, 1> -> f64
    %w = arith.mulf %dt, %a : f64
    ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// CHECK-LABEL: ops.chain @chain
// CHECK-SAME: [[G:%[a-z0-9]+]]: f64, [[DT:%[a-z0-9]+]]: f64)
// CHECK-SAME: ops.binding_consts = [{arg = 7 : i64, name = "dt", offset = 0 : i64}]
// CHECK: ops.par_loop @guarded
// CHECK-SAME: scalars([[G]], [[DT]] : f64, f64)
// CHECK: ^bb0({{.*}}, [[LG:%[a-z0-9]+]]: f64, [[LDT:%[a-z0-9]+]]: f64):
// CHECK: scf.if
// CHECK-NOT: ops.const_ref
// CHECK-NOT: ops.gbl_read
// CHECK: arith.mulf [[LDT]], [[LG]]
ops.chain @chain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index,
                 %hi1: index, %g: f64) {
  ops.par_loop @guarded range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars(%g : f64) {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %s: f64):
  }
  ops.return
}
