// A conditional expression that loads from two different dats (`a < b ? a[i+1] : b[i+1]`) lowers to a cond_br whose two
// successors are identical up to the memref they load from; branch canonicalisation after the conversion to the LLVM
// dialect used to merge them into one block with memref arguments, which the LLVM dialect cannot have (CloverLeaf 3D
// calc_dt_kernel).
// RUN: ops-opt %s -ops-jit-cpu="omp=false" | FileCheck %s

// CHECK: llvm.func @cond
// CHECK-NOT: memref<
ops.chain @cond_group(%lo0: index, %hi0: index, %a: !ops.dat<f64, 1, 1>, %b: !ops.dat<f64, 1, 1>, %c: !ops.dat<f64, 1, 1>)
    attributes {ops.func_name = "cond"} {
  ops.par_loop range(%lo0, %hi0)
      dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0], [1]]>, %b : !ops.dat<f64, 1, 1> read #ops.stencil<[[0], [1]]>,
           %c : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>, %z: !ops.acc<f64, 1, 1>):
    %p = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
    %q = ops.read %y[0] : !ops.acc<f64, 1, 1> -> f64
    %lt = arith.cmpf olt, %p, %q : f64
    %m = scf.if %lt -> (f64) {
      %u = ops.read %x[1] : !ops.acc<f64, 1, 1> -> f64
      scf.yield %u : f64
    } else {
      %w = ops.read %y[1] : !ops.acc<f64, 1, 1> -> f64
      scf.yield %w : f64
    }
    ops.write %z[0], %m : !ops.acc<f64, 1, 1>
  }
  ops.return
}
