// The CPU pipeline down to LLVM IR: the dat arguments are noalias pointers and LLVM vectorises a point-wise group
// without run-time overlap checks.
// REQUIRES: x86
// RUN: ops-opt %s -ops-jit-cpu="omp=false" | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: FileCheck %s --check-prefix=NOALIAS < %t.ll
// RUN: opt -S -passes='default<O3>' -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 %t.ll | FileCheck %s --check-prefix=AVX2

// NOALIAS: define void @group(ptr %{{.*}}, ptr noalias %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, ptr noalias %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, ptr noalias
// AVX2-NOT: memcheck
// AVX2: vector.body:
// AVX2: load <4 x double>
// AVX2: load <4 x double>
// AVX2: fadd <4 x double>
// AVX2: store <4 x double>
ops.chain @group(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 1>,
                 %c: !ops.dat<f64, 2, 1>) {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>,
           %c : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %z: !ops.acc<f64, 2, 1>):
    %u = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %v = ops.read %y[0, 0] : !ops.acc<f64, 2, 1> -> f64
    %s = arith.addf %u, %v : f64
    ops.write %z[0, 0], %s : !ops.acc<f64, 2, 1>
  }
  ops.return
}
