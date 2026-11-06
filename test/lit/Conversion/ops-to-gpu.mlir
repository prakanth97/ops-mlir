// convert-ops-to-scf target=gpu: the host side of a kernel launch: grid from the range, one thread per point, a mask
// for the threads outside the range, a shuffle reduction per warp and one store per warp into a device buffer argument
// per reduction (one element per warp; the caller combines them); nothing is returned.
// RUN: ops-opt %s -convert-ops-to-scf="target=gpu block=32,4,1" | FileCheck %s

// CHECK-LABEL: func.func @sum_max(
// CHECK-SAME: %[[A:arg0]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[B:arg1]]: memref<?x?xf64, strided<[?, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[LO0:arg6]]: index, %[[HI0:arg7]]: index, %[[LO1:arg8]]: index, %[[HI1:arg9]]: index
// CHECK-SAME: %[[BUF:arg12]]: memref<?xf64>)
// CHECK-SAME: gpu = true
// CHECK-NOT: -> f64
// CHECK: arith.ceildivsi %{{.*}}, %[[C32:.*]] : index
// CHECK: arith.ceildivsi %{{.*}}, %[[C4:.*]] : index
// CHECK: scf.if %[[NONEMPTY:.*]] {
// CHECK: gpu.launch blocks(%[[BX:[^,)]*]], %[[BY:[^,)]*]], %{{[^,)]*}}) in
// CHECK-SAME: threads(%[[TX:[^,)]*]], %[[TY:[^,)]*]], %{{[^,)]*}}) in
// CHECK: arith.cmpi slt, %{{.*}}, %[[HI0]]
// CHECK: arith.cmpi slt, %{{.*}}, %[[HI1]]
// CHECK: memref.alloca() {ops.accumulator} : memref<f64>
// CHECK: scf.if %{{.*}} {
// CHECK: memref.load %[[A]]
// CHECK: memref.store %{{.*}}, %[[B]]
// CHECK: gpu.shuffle xor %{{.*}}, %{{.*}}, %{{.*}} : f64
// CHECK: scf.if %{{.*}} {
// CHECK: memref.store %{{.*}}, %[[BUF]][%{{.*}}] : memref<?xf64>
// CHECK: gpu.terminator
ops.chain @sum_max(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %w: !ops.dat<f64, 2, 1>, %o: !ops.dat<f64, 2, 1>) -> f64 {
  %s = ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%w : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %o : !ops.dat<f64, 2, 1> write #ops.stencil<[[0, 0]]>) scalars()
      attributes {reduction_kinds = [#ops.red_kind<max>]} -> f64 {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 1>, %acc: !ops.red<f64>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %v : !ops.acc<f64, 2, 1>
    ops.reduce %acc, %v : !ops.red<f64>
  }
  ops.return %s : f64
}
