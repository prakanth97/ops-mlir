// convert-ops-to-scf: the compiled group ABI.
// RUN: ops-opt %s -split-input-file -convert-ops-to-scf | FileCheck %s
// RUN: ops-opt %s -split-input-file -convert-ops-to-scf=soa=true | FileCheck %s --check-prefix=SOA

// Point-wise, 2D: dats in order of first appearance, shifts per dat and dimension, range, displacements, scalars.
// CHECK-LABEL: func.func @pointwise(
// CHECK-SAME: %[[A:arg0]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[B:arg1]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[SA0:arg2]]: index, %[[SA1:arg3]]: index, %[[SB0:arg4]]: index, %[[SB1:arg5]]: index
// CHECK-SAME: %[[LO0:arg6]]: index, %[[HI0:arg7]]: index, %[[LO1:arg8]]: index, %[[HI1:arg9]]: index
// CHECK-SAME: %[[D0:arg10]]: index, %[[D1:arg11]]: index
// CHECK-SAME: %[[S:arg12]]: f64
// CHECK-SAME: attributes {llvm.emit_c_interface, ops.abi = {gpu = false, noalias_args = array<i64: 1, 8>, num_dats = 2 : i64, num_reductions = 0 : i64, num_scalars = 1 : i64, rank = 2 : i64, soa = false}}
// CHECK: scf.parallel (%[[J:.*]]) = (%[[LO1]]) to (%[[HI1]])
// CHECK: scf.for %[[I:.*]] = %[[LO0]] to %[[HI0]]
// CHECK: %[[IA0:.*]] = arith.addi %[[SA0]], %[[I]]
// CHECK: %[[IA1:.*]] = arith.addi %[[SA1]], %[[J]]
// CHECK: %[[V:.*]] = memref.load %[[A]][%[[IA1]], %[[IA0]]]
// CHECK: %[[W:.*]] = arith.mulf %[[V]], %[[S]]
// CHECK: %[[IB0:.*]] = arith.addi %[[SB0]], %[[I]]
// CHECK: %[[IB1:.*]] = arith.addi %[[SB1]], %[[J]]
// CHECK: memref.store %[[W]], %[[B]][%[[IB1]], %[[IB0]]]
ops.chain @pointwise(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %a: !ops.dat<f64, 2, 1>,
                     %b: !ops.dat<f64, 2, 1>, %s: f64) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars(%s : f64) {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %sc: f64):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %w = arith.mulf %v, %sc : f64
    ops.write %y[0, 0], %w : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// -----

// Stencil offsets are added to the memref index; a write at a non-zero offset; a read of a dat that lacks dimension y
// (stride 0) does not use the y coordinate; ops.index is the coordinate plus the displacement of the block.
// CHECK-LABEL: func.func @stencil(
// CHECK-SAME: %[[U:arg0]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[X:arg1]]: memref<?x?xi32, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[SU0:arg2]]: index, %[[SU1:arg3]]: index, %[[SX0:arg4]]: index, %[[SX1:arg5]]: index
// CHECK-SAME: %[[D0:arg10]]: index, %[[D1:arg11]]: index
// CHECK: scf.parallel (%[[J:.*]]) =
// CHECK: scf.for %[[I:.*]] =
// CHECK: arith.addi %[[SU0]], %[[I]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: arith.addi %{{.*}}, %[[C1]]
// CHECK: memref.load %[[U]]
// CHECK: arith.addi %[[SX0]], %[[I]]
// CHECK-NOT: arith.addi %[[SX1]], %[[J]]
// CHECK: memref.load %[[X]][%[[SX1]], %{{.*}}]
// CHECK: arith.addi %[[I]], %[[D0]]
// CHECK: arith.index_cast %{{.*}} : index to i32
// CHECK: %[[C_1:.*]] = arith.constant -1 : index
// CHECK: memref.store %{{.*}}, %[[U]]
ops.chain @stencil(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %u: !ops.dat<f64, 2, 1>,
                   %xs: !ops.dat<i32, 2, 1>) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> rw #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>,
           %xs : !ops.dat<i32, 2, 1> read #ops.stencil<[[0, 0]], strides = [1, 0]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %c: !ops.acc<i32, 2, 1>):
    %r = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %k = ops.read %c[0, 0] : !ops.acc<i32, 2, 1> -> i32
    %i = ops.index 0
    %sum = arith.addi %k, %i : i32
    %f = arith.sitofp %sum : i32 to f64
    %z = arith.addf %r, %f : f64
    ops.write %x[-1, 0], %z : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// -----

// Two components. Array of structures: the components are adjacent (strides [?, 2, 1]); structure of arrays:
// [?, 1, ?]. The component is the last memref index either way.
// CHECK-LABEL: func.func @multicomp(
// CHECK-SAME: memref<?x?x2xf32, strided<[?, 2, 1], offset: ?>> {llvm.noalias}
// CHECK: scf.parallel
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: memref.load %arg0[%{{.*}}, %{{.*}}, %[[C1]]]
// SOA-LABEL: func.func @multicomp(
// SOA-SAME: memref<?x?x2xf32, strided<[?, 1, ?], offset: ?>> {llvm.noalias}
// SOA: ops.abi = {{.*}}soa = true
ops.chain @multicomp(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %w: !ops.dat<f32, 2, 2>) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%w : !ops.dat<f32, 2, 2> rw #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f32, 2, 2>):
    %v = ops.read %x[0, 0] {comp = 1 : i64} : !ops.acc<f32, 2, 2> -> f32
    ops.write %x[0, 0], %v : !ops.acc<f32, 2, 2>
  }
  ops.return
}

// -----

// Reductions: a per-point accumulator folded with the kind, one scf.reduce; the function returns the partial values.
// CHECK-LABEL: func.func @reductions(
// CHECK-SAME: -> (f64, i32)
// CHECK: %[[INF:.*]] = arith.constant 0x7FF0000000000000 : f64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[R:.*]]:2 = scf.parallel
// CHECK-SAME: init (%[[INF]], %[[ZERO]]) -> (f64, i32)
// CHECK: memref.alloca() {ops.accumulator} : memref<f64>
// CHECK: memref.alloca() {ops.accumulator} : memref<i32>
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.addi
// CHECK: scf.reduce(%{{.*}}, %{{.*}} : f64, i32)
// CHECK: arith.cmpf olt
// CHECK: scf.reduce.return
// CHECK: arith.addi
// CHECK: scf.reduce.return
// CHECK: return %[[R]]#0, %[[R]]#1 : f64, i32
ops.chain @reductions(%lo0: index, %hi0: index, %w: !ops.dat<f64, 1, 1>) -> (f64, i32) {
  %m, %n = ops.par_loop range(%lo0, %hi0)
      dats(%w : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<min>, #ops.red_kind<add>]} -> f64, i32 {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %mn: !ops.red<f64>, %cnt: !ops.red<i32>):
    %v = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
    ops.reduce %mn, %v : !ops.red<f64>
    %one = arith.constant 1 : i32
    ops.reduce %cnt, %one : !ops.red<i32>
  }
  ops.return %m, %n : f64, i32
}
