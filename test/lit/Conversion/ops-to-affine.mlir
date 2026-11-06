// convert-ops-to-scf loops=affine: affine.parallel / affine.for and affine accesses, for the transformations that work on affine loops.
// RUN: ops-opt %s -split-input-file -convert-ops-to-scf=loops=affine | FileCheck %s
// RUN: ops-opt %s -split-input-file -convert-ops-to-scf=loops=affine -lower-affine | FileCheck %s --check-prefix=LOWER
// RUN: not ops-opt %s -convert-ops-to-scf=loops=poly 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: ops-opt %s -split-input-file -pass-pipeline='builtin.module(convert-ops-to-scf{loops=affine},func.func(affine-loop-unroll{unroll-factor=2}))' | FileCheck %s --check-prefix=UNROLL

// Point-wise, 2D: the outer dimension is affine.parallel, x is affine.for, the accesses are shift + coordinate.
// CHECK-LABEL: func.func @pointwise(
// CHECK-SAME: %[[A:arg0]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[B:arg1]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[SA0:arg2]]: index, %[[SA1:arg3]]: index, %[[SB0:arg4]]: index, %[[SB1:arg5]]: index
// CHECK-SAME: %[[LO0:arg6]]: index, %[[HI0:arg7]]: index, %[[LO1:arg8]]: index, %[[HI1:arg9]]: index
// CHECK-SAME: %[[S:arg12]]: f64
// CHECK-SAME: ops.loops = "affine"
// CHECK: affine.parallel (%[[J:.*]]) = (symbol(%[[LO1]])) to (symbol(%[[HI1]]))
// CHECK: affine.for %[[I:.*]] = %[[LO0]] to %[[HI0]]
// CHECK: %[[V:.*]] = affine.load %[[A]][%[[J]] + symbol(%[[SA1]]), %[[I]] + symbol(%[[SA0]])]
// CHECK: %[[W:.*]] = arith.mulf %[[V]], %[[S]]
// CHECK: affine.store %[[W]], %[[B]][%[[J]] + symbol(%[[SB1]]), %[[I]] + symbol(%[[SB0]])]
// CHECK-NOT: memref.load
// UNROLL-LABEL: func.func @pointwise(
// UNROLL: affine.parallel
// UNROLL: affine.for {{.*}} step 2
// LOWER-LABEL: func.func @pointwise(
// LOWER: scf.parallel
// LOWER: scf.for
// LOWER-NOT: affine.
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

// Stencil offsets are constants of the map; a dat without dimension y (stride 0) does not use the y coordinate.
// CHECK-LABEL: func.func @stencil(
// CHECK-SAME: %[[U:arg0]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[X:arg1]]: memref<?x?xi32, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[SU0:arg2]]: index, %[[SU1:arg3]]: index, %[[SX0:arg4]]: index, %[[SX1:arg5]]: index
// CHECK: affine.parallel (%[[J:.*]]) =
// CHECK: affine.for %[[I:.*]] =
// CHECK: affine.load %[[U]][%[[J]] + symbol(%[[SU1]]), %[[I]] + symbol(%[[SU0]]) + 1]
// CHECK: affine.load %[[X]][symbol(%[[SX1]]), %[[I]] + symbol(%[[SX0]])]
// CHECK: affine.store %{{.*}}, %[[U]][%[[J]] + symbol(%[[SU1]]), %[[I]] + symbol(%[[SU0]]) - 1]
ops.chain @stencil(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %u: !ops.dat<f64, 2, 1>,
                   %xs: !ops.dat<i32, 2, 1>) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> rw #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>,
           %xs : !ops.dat<i32, 2, 1> read #ops.stencil<[[0, 0]], strides = [1, 0]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %c: !ops.acc<i32, 2, 1>):
    %r = ops.read %x[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %k = ops.read %c[0, 0] : !ops.acc<i32, 2, 1> -> i32
    %f = arith.sitofp %k : i32 to f64
    %z = arith.addf %r, %f : f64
    ops.write %x[-1, 0], %z : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// -----

// 3D: two parallel dimensions (z, y) in one affine.parallel, x the affine.for. (Every bound map of an affine.parallel sees all
// the bound operands: two parallel dimensions need two symbols per map.)
// CHECK-LABEL: func.func @threed(
// CHECK-SAME: %[[A:arg0]]: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[LO0:arg4]]: index, %[[HI0:arg5]]: index, %[[LO1:arg6]]: index, %[[HI1:arg7]]: index, %[[LO2:arg8]]: index, %[[HI2:arg9]]: index
// CHECK: affine.parallel (%[[K:.*]], %[[J:.*]]) = (symbol(%[[LO2]]), symbol(%[[LO1]])) to (symbol(%[[HI2]]), symbol(%[[HI1]]))
// CHECK: affine.for %[[I:.*]] = %[[LO0]] to %[[HI0]]
// CHECK: affine.load %[[A]][%[[K]] + symbol(%{{.*}}), %[[J]] + symbol(%{{.*}}), %[[I]] + symbol(%{{.*}}) + 1]
// CHECK: affine.store %{{.*}}, %[[A]][%[[K]] + symbol(%{{.*}}), %[[J]] + symbol(%{{.*}}), %[[I]] + symbol(%{{.*}})]
ops.chain @threed(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %lo2: index, %hi2: index, %u: !ops.dat<f64, 3, 1>) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1, %lo2, %hi2)
      dats(%u : !ops.dat<f64, 3, 1> rw #ops.stencil<[[0, 0, 0], [1, 0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 3, 1>):
    %v = ops.read %x[1, 0, 0] : !ops.acc<f64, 3, 1> -> f64
    ops.write %x[0, 0, 0], %v : !ops.acc<f64, 3, 1>
  }
  ops.return
}

// -----

// A data-dependent offset is not affine: it stays a memref.load next to the affine accesses.
// CHECK-LABEL: func.func @dynamic(
// CHECK: affine.parallel
// CHECK: affine.for
// CHECK: affine.load
// CHECK: memref.load
// CHECK: affine.store
ops.chain @dynamic(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %u: !ops.dat<f64, 2, 1>, %v: !ops.dat<f64, 2, 1>) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0], [1, 0], [-1, 0]]>, %v : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>):
    %a = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %c0 = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %pos = arith.cmpf ogt, %a, %a : f64
    %off = arith.select %pos, %one, %c0 : i32
    %d = ops.read_dyn %x[%off, %c0 : i32, i32] : !ops.acc<f64, 2, 1> -> f64
    %s = arith.addf %a, %d : f64
    ops.write %y[0, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// -----

// A loop with a reduction keeps the scf form (and says so).
// CHECK-LABEL: func.func @reduction(
// CHECK-SAME: ops.loops = "scf"
// CHECK: scf.parallel
// CHECK-NOT: affine.parallel
ops.chain @reduction(%lo0: index, %hi0: index, %w: !ops.dat<f64, 1, 1>) -> f64 {
  %r = ops.par_loop range(%lo0, %hi0)
      dats(%w : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<add>]} -> f64 {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %red: !ops.red<f64>):
    %v = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
    ops.reduce %red, %v : !ops.red<f64>
  }
  ops.return %r : f64
}

// BAD: option 'loops' must be scf or affine, got 'poly'
