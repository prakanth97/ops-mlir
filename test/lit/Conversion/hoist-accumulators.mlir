// The accumulator of a reduction is initialised per iteration of the parallel loop but allocated once per thread
// (the entry of the omp.parallel region) or, sequentially, once per call (the function entry).
// RUN: ops-opt %s -convert-ops-to-scf -convert-scf-to-openmp -ops-hoist-accumulators | FileCheck %s --check-prefix=OMP
// RUN: ops-opt %s -convert-ops-to-scf -ops-hoist-accumulators | FileCheck %s --check-prefix=SEQ

// OMP-LABEL: func.func @sum
// OMP: omp.parallel {
// OMP-NEXT: memref.alloca() : memref<f64>
// OMP: omp.wsloop
// OMP-NOT: memref.alloca()

// SEQ-LABEL: func.func @sum
// SEQ: memref.alloca() : memref<f64>
// SEQ: scf.parallel
// SEQ-NOT: memref.alloca()
ops.chain @sum(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %w: !ops.dat<f64, 2, 1>) -> f64 {
  %s = ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%w : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<add>]} -> f64 {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %acc: !ops.red<f64>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.reduce %acc, %v : !ops.red<f64>
  }
  ops.return %s : f64
}
