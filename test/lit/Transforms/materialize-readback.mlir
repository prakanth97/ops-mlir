// A kernel may read back what it wrote to its own (write-only) output: the read is replaced by the stored value.
// RUN: ops-opt %s -ops-materialize-kernels | FileCheck %s

ops.kernel @laplace_error(%a: !ops.acc<f64, 1, 1>, %anew: !ops.acc<f64, 1, 1>) {
  %v = ops.read %a[0] : !ops.acc<f64, 1, 1> -> f64
  %two = arith.constant 2.0 : f64
  %w = arith.mulf %v, %two : f64
  ops.write %anew[0], %w : !ops.acc<f64, 1, 1>
  %back = ops.read %anew[0] : !ops.acc<f64, 1, 1> -> f64
  %d = arith.subf %back, %v : f64
  ops.write %a[0], %d : !ops.acc<f64, 1, 1>
  ops.return
}

// CHECK-LABEL: ops.chain @c
// CHECK: ops.par_loop @laplace_error
// CHECK: ^bb0([[A:%[a-z0-9]+]]: !ops.acc<f64, 1, 1>, [[N:%[a-z0-9]+]]: !ops.acc<f64, 1, 1>):
// CHECK-NEXT: [[V:%[0-9]+]] = ops.read [[A]] [0]
// CHECK-NOT: ops.read [[N]]
// CHECK: [[W:%[0-9]+]] = arith.mulf [[V]]
// CHECK-NEXT: ops.write [[N]] [0], [[W]]
// CHECK-NEXT: arith.subf [[W]], [[V]]
ops.chain @c(%a: !ops.dat<f64, 1, 1>, %b: !ops.dat<f64, 1, 1>, %lo: index, %hi: index) {
  ops.par_loop @laplace_error range(%lo, %hi)
      dats(%a : !ops.dat<f64, 1, 1> rw #ops.stencil<[[0]]>, %b : !ops.dat<f64, 1, 1> write #ops.stencil<[[0]]>) scalars() {
  ^bb0(%x: !ops.acc<f64, 1, 1>, %y: !ops.acc<f64, 1, 1>):
  }
  ops.return
}
