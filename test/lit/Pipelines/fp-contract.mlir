// `fma=true` sets the contract fast-math flag on the floating point operations (the code generator may then fuse a
// multiply and an add); by default the operations are untouched.
// RUN: ops-opt %s -ops-jit-cpu="omp=false fma=true" | FileCheck %s --check-prefix=FMA
// RUN: ops-opt %s -ops-jit-cpu="omp=false" | FileCheck %s --check-prefix=PLAIN

// FMA: llvm.fmul {{.*}}#llvm.fastmath<contract>
// FMA: llvm.fadd {{.*}}#llvm.fastmath<contract>
// PLAIN: llvm.fmul %{{[0-9a-z]+}}, %{{[0-9a-z]+}} : f64
// PLAIN-NOT: contract
ops.chain @g(%lo0: index, %hi0: index, %a: !ops.dat<f64, 1, 1>, %b: !ops.dat<f64, 1, 1>, %s: f64)
    attributes {ops.func_name = "g"} {
  ops.par_loop range(%lo0, %hi0)
      dats(%a : !ops.dat<f64, 1, 1> read #ops.stencil<[[0]]>, %b : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars(%s : f64) {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>, %c: f64):
    %v = ops.read %x[0] : !ops.acc<f64, 1, 1> -> f64
    %m = arith.mulf %v, %c : f64
    %r = arith.addf %m, %v : f64
    ops.write %y[0], %r : !ops.acc<f64, 1, 1>
  }
  ops.return
}
