// Executes lowered groups with mlir-runner on hand-filled memrefs: stencil offsets, shifts, scalars, the block
// displacement and reductions. Through the full CPU pipeline, sequential and with OpenMP.
// RUN: ops-opt %s -ops-jit-cpu="omp=false" \
// RUN:   | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:   | FileCheck %s
// RUN: ops-opt %s -ops-jit-cpu="omp=true" \
// RUN:   | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%mlir_omp \
// RUN:   | FileCheck %s
// RUN: ops-opt %s -ops-jit-cpu="omp=false specialize=ranges" \
// RUN:   | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:   | FileCheck %s

// v(i, j) = u(i+1, j) + u(i, j+1) over [1,4) x [1,4); the dats are 6 x 6 with zero shifts.
ops.chain @stencil_group(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %u: !ops.dat<f64, 2, 1>,
                         %v: !ops.dat<f64, 2, 1>) attributes {ops.func_name = "stencil", ops.range_values = [array<i64: 1, 4, 1, 4>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> read #ops.stencil<[[1, 0], [0, 1]]>,
           %v : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%a: !ops.acc<f64, 2, 1>, %b: !ops.acc<f64, 2, 1>):
    %r = ops.read %a[1, 0] : !ops.acc<f64, 2, 1> -> f64
    %t = ops.read %a[0, 1] : !ops.acc<f64, 2, 1> -> f64
    %s = arith.addf %r, %t : f64
    ops.write %b[0, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// v = u * scale + (global x index); the block starts at x = 100.
ops.chain @index_group(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %u: !ops.dat<f64, 2, 1>,
                       %v: !ops.dat<f64, 2, 1>, %scale: f64) attributes {ops.func_name = "indexed", ops.range_values = [array<i64: 1, 4, 1, 4>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %v : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>)
      scalars(%scale : f64) {
  ^bb0(%a: !ops.acc<f64, 2, 1>, %b: !ops.acc<f64, 2, 1>, %sc: f64):
    %x = ops.read %a[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %m = arith.mulf %x, %sc : f64
    %i = ops.index 0
    %f = arith.sitofp %i : i32 to f64
    %s = arith.addf %m, %f : f64
    ops.write %b[0, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.return
}

// min and sum of u over the range.
ops.chain @reduce_group(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %u: !ops.dat<f64, 2, 1>) -> (f64, f64)
    attributes {ops.func_name = "reduced", ops.range_values = [array<i64: 1, 4, 1, 4>]} {
  %mn, %sum = ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%u : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<min>, #ops.red_kind<add>]} -> f64, f64 {
  ^bb0(%a: !ops.acc<f64, 2, 1>, %m: !ops.red<f64>, %s: !ops.red<f64>):
    %x = ops.read %a[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.reduce %m, %x : !ops.red<f64>
    ops.reduce %s, %x : !ops.red<f64>
  }
  ops.return %mn, %sum : f64, f64
}

// The declarations the callers use; the lowering replaces them by the definitions.
func.func private @stencil(memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>,
                           index, index, index, index, index, index, index, index, index, index)
func.func private @indexed(memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>,
                           index, index, index, index, index, index, index, index, index, index, f64)
func.func private @reduced(memref<?x?xf64, strided<[?, 1], offset: ?>>, index, index, index, index, index, index,
                           index, index) -> (f64, f64)
func.func private @printMemrefF64(memref<*xf64>)
func.func private @printF64(f64)
func.func private @printNewline()

// m[j][i] = 10 j + i, or zero.
func.func @fill(%m: memref<6x6xf64>, %scale: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %ten = arith.constant 10 : i64
  scf.for %j = %c0 to %c6 step %c1 {
    scf.for %i = %c0 to %c6 step %c1 {
      %jj = arith.index_cast %j : index to i64
      %ii = arith.index_cast %i : index to i64
      %t = arith.muli %jj, %ten : i64
      %s = arith.addi %t, %ii : i64
      %v = arith.sitofp %s : i64 to f64
      %w = arith.mulf %v, %scale : f64
      memref.store %w, %m[%j, %i] : memref<6x6xf64>
    }
  }
  return
}

func.func @show(%m: memref<6x6xf64>) {
  %u = memref.cast %m : memref<6x6xf64> to memref<*xf64>
  call @printMemrefF64(%u) : (memref<*xf64>) -> ()
  return
}

func.func @main() {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  %u = memref.alloc() : memref<6x6xf64>
  %v = memref.alloc() : memref<6x6xf64>
  call @fill(%u, %one) : (memref<6x6xf64>, f64) -> ()
  call @fill(%v, %zero) : (memref<6x6xf64>, f64) -> ()
  %ud = memref.cast %u : memref<6x6xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
  %vd = memref.cast %v : memref<6x6xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>

  // No shifts. Rows 1..3, columns 1..3 of v: u(i+1, j) + u(i, j+1) = 20 j + 2 i + 11.
  call @stencil(%ud, %vd, %c0, %c0, %c0, %c0, %c1, %c4, %c1, %c4, %c0, %c0)
      : (memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, index, index, index,
         index, index, index, index, index, index, index) -> ()
  call @show(%v) : (memref<6x6xf64>) -> ()
  // CHECK: [0, 0, 0, 0, 0, 0],
  // CHECK-NEXT: [0, 33, 35, 37, 0, 0],
  // CHECK-NEXT: [0, 53, 55, 57, 0, 0],
  // CHECK-NEXT: [0, 73, 75, 77, 0, 0],
  // CHECK-NEXT: [0, 0, 0, 0, 0, 0],

  // A shift of one in x for u moves what is read one column to the right: 20 j + 2 i + 13.
  call @fill(%v, %zero) : (memref<6x6xf64>, f64) -> ()
  call @stencil(%ud, %vd, %c1, %c0, %c0, %c0, %c1, %c4, %c1, %c4, %c0, %c0)
      : (memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, index, index, index,
         index, index, index, index, index, index, index) -> ()
  call @show(%v) : (memref<6x6xf64>) -> ()
  // CHECK: [0, 35, 37, 39, 0, 0],
  // CHECK-NEXT: [0, 55, 57, 59, 0, 0],
  // CHECK-NEXT: [0, 75, 77, 79, 0, 0],

  // Scalar and displacement: u * 2 + (i + 100).
  call @fill(%v, %zero) : (memref<6x6xf64>, f64) -> ()
  call @indexed(%ud, %vd, %c0, %c0, %c0, %c0, %c1, %c4, %c1, %c4, %c100, %c0, %two)
      : (memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, index, index, index,
         index, index, index, index, index, index, index, f64) -> ()
  call @show(%v) : (memref<6x6xf64>) -> ()
  // CHECK: [0, 123, 126, 129, 0, 0],
  // CHECK-NEXT: [0, 143, 146, 149, 0, 0],
  // CHECK-NEXT: [0, 163, 166, 169, 0, 0],

  // Reductions over u in [1,4) x [1,4): the minimum is u(1, 1) = 11, the sum 198.
  %mn, %sum = call @reduced(%ud, %c0, %c0, %c1, %c4, %c1, %c4, %c0, %c0)
      : (memref<?x?xf64, strided<[?, 1], offset: ?>>, index, index, index, index, index, index, index, index) -> (f64, f64)
  call @printF64(%mn) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%sum) : (f64) -> ()
  call @printNewline() : () -> ()
  // CHECK: 11
  // CHECK-NEXT: 198
  return
}
