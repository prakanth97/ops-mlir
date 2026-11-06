// ops-forward-stores: a value written to a point is used instead of reading it back; access modes narrow.
// RUN: ops-opt %s -ops-forward-stores | FileCheck %s

// A write followed by a read of the same point: forwarded, and the operand no longer needs to be loaded.
// CHECK-LABEL: ops.chain @forward
// CHECK: dats(%arg0 : !ops.dat<f64, 2, 1> write
// CHECK: ^bb0([[X:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>, [[Y:%[a-z0-9]+]]: !ops.acc<f64, 2, 1>):
// CHECK-NOT: ops.read [[X]]
// CHECK: [[RD:%[0-9]+]] = ops.read [[Y]] [1, 0]
// CHECK: arith.addf [[V:%[a-z0-9]+]], [[RD]]
ops.chain @forward(%a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>, %lo: index, %hi: index, %lo1: index, %hi1: index)
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>]} {
  ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> rw #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %v = ops.read %y[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %x[0, 0], %v : !ops.acc<f64, 2, 1>
    %back = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %n = ops.read %y[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %s = arith.addf %back, %n : f64
    ops.write %x[0, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// A different point or component is not forwarded; a nested region that writes the accessor forgets the value; an
// operand that is read and written stays rw.
// CHECK-LABEL: ops.chain @not_forwarded
// CHECK: %arg0 : !ops.dat<f64, 2, 2> rw
// CHECK: ops.read {{.*}} [1, 0]
// CHECK: ops.read {{.*}} {comp = 1 : i64}
// CHECK: scf.if
// CHECK: ops.read
ops.chain @not_forwarded(%a: !ops.dat<f64, 2, 2>, %c: i1, %lo: index, %hi: index, %lo1: index, %hi1: index)
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>]} {
  ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 2> rw #ops.stencil<[[0, 0], [1, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 2>):
    %z = arith.constant 1.0 : f64
    ops.write %x[0, 0], %z : !ops.acc<f64, 2, 2>
    %other_point = ops.read %x[1, 0] : !ops.acc<f64, 2, 2> -> f64
    %other_comp = ops.read %x[0, 0] {comp = 1 : i64} : !ops.acc<f64, 2, 2> -> f64
    scf.if %c {
      ops.write %x[0, 0], %other_point : !ops.acc<f64, 2, 2>
    }
    %after_region = ops.read %x[0, 0] : !ops.acc<f64, 2, 2> -> f64
    %s = arith.addf %other_comp, %after_region : f64
    ops.write %x[0, 0], %s {comp = 1 : i64} : !ops.acc<f64, 2, 2>
  }
  ops.return
}

// The same dat through two operands: not touched.
// CHECK-LABEL: ops.chain @aliased
// CHECK: %arg0 : !ops.dat<f64, 2, 1> rw
// CHECK-SAME: %arg0 : !ops.dat<f64, 2, 1> read
// CHECK: ops.read
ops.chain @aliased(%a: !ops.dat<f64, 2, 1>, %lo: index, %hi: index, %lo1: index, %hi1: index)
    attributes {"ops.range_values" = [array<i64: 0, 8, 0, 8>]} {
  ops.par_loop range(%lo, %hi, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> rw #ops.stencil<[[0, 0]]>, %a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %z = arith.constant 1.0 : f64
    ops.write %x[0, 0], %z : !ops.acc<f64, 2, 1>
    %v = ops.read %y[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %x[0, 0], %v : !ops.acc<f64, 2, 1>
  }
  ops.return
}
