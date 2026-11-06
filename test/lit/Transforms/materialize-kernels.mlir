// ops-materialize-kernels: kernel bodies cloned into the loops that call them.
// RUN: ops-opt %s -ops-materialize-kernels | FileCheck %s
// RUN: ops-opt %s -ops-materialize-kernels | ops-opt -ops-materialize-kernels | FileCheck %s

ops.kernel @copy(%in: !ops.acc<f64, 2, 1>, %out: !ops.acc<f64, 2, 1>) {
  %v = ops.read %in[0, 0] : !ops.acc<f64, 2, 1> -> f64
  ops.write %out[0, 0], %v : !ops.acc<f64, 2, 1>
  ops.return
}

// Global components become scalar block arguments, in the order the loop passes them; ops.index stays.
ops.kernel @axpy(%x: !ops.acc<f64, 2, 1>, %g: !ops.gbl<f64, 2>, %y: !ops.acc<f64, 2, 1>) {
  %a = ops.gbl_read %g[1] : !ops.gbl<f64, 2> -> f64
  %b = ops.gbl_read %g[0] : !ops.gbl<f64, 2> -> f64
  %i = ops.index 0
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %p = arith.mulf %v, %a : f64
  %q = arith.addf %p, %b : f64
  ops.write %y[0, 0], %q : !ops.acc<f64, 2, 1>
  ops.return
}

// Constants: one chain argument per (name, offset, type), shared by the loops that use it.
ops.kernel @scaled(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>) {
  %dt = ops.const_ref @dt : f64
  %n = ops.const_ref @grid {offset = 8 : i64} : i32
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  %w = arith.mulf %v, %dt : f64
  ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  ops.return
}

// A reduction parameter becomes the loop's reduction argument.
ops.kernel @sum(%x: !ops.acc<f64, 2, 1>, %r: !ops.red<f64>) {
  %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
  ops.reduce %r, %v : !ops.red<f64>
  ops.return
}

// CHECK-LABEL: ops.chain @chain
// CHECK-SAME: [[G0:%[a-z0-9]+]]: f64, [[G1:%[a-z0-9]+]]: f64, [[DT:%[a-z0-9]+]]: f64, [[N:%[a-z0-9]+]]: i32) -> f64
// CHECK-SAME: ops.binding_consts = [{arg = 8 : i64, name = "dt", offset = 0 : i64}, {arg = 9 : i64, name = "grid", offset = 8 : i64}]
// CHECK: ops.par_loop @copy
// CHECK: ^bb0([[IN:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>, [[OUT:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>):
// CHECK-NEXT: [[V:%[0-9]+]] = ops.read [[IN]] [0, 0]
// CHECK-NEXT: ops.write [[OUT]] [0, 0], [[V]]
// CHECK: ops.par_loop @axpy
// CHECK-SAME: scalars([[G0]], [[G1]] : f64, f64)
// CHECK: ^bb0({{.*}}, [[S0:%[a-z0-9]+]]: f64, [[S1:%[a-z0-9]+]]: f64):
// CHECK: ops.index 0
// CHECK: arith.mulf {{.*}}, [[S1]]
// CHECK: arith.addf {{.*}}, [[S0]]
// CHECK: ops.par_loop @scaled
// CHECK-SAME: scalars([[DT]], [[N]] : f64, i32)
// CHECK: ^bb0({{.*}}, [[LDT:%[a-z0-9]+]]: f64, {{.*}}: i32):
// CHECK: arith.mulf {{.*}}, [[LDT]]
// CHECK: ops.par_loop @scaled
// CHECK-SAME: scalars([[DT]], [[N]] : f64, i32)
// CHECK: ops.par_loop @sum
// CHECK: ^bb0({{.*}}, [[R:%[a-z0-9]+]]: !ops.red<f64>):
// CHECK: ops.reduce [[R]]
ops.chain @chain(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo0: index, %hi0: index, %lo1: index,
                 %hi1: index, %g0: f64, %g1: f64) -> f64 {
  ops.par_loop @copy range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop @axpy range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars(%g0, %g1 : f64, f64) {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %s0: f64, %s1: f64):
  }
  ops.par_loop @scaled range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  ops.par_loop @scaled range(%lo0, %hi0, %lo1, %hi1)
      dats(%b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %a : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
  }
  %r = ops.par_loop @sum range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<add>]} -> f64 {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %red: !ops.red<f64>):
  }
  ops.return %r : f64
}
