// ops-specialize mode=shapes: the sizes of the dats (chain attribute ops.shape_values, per argument, x first) make the memref
// types of the group function static (constant strides), the offset stays dynamic; a multi-component dat with
// interleaved components has a constant component stride, and the structure-of-arrays layout a constant stride of
// the number of points.
// RUN: ops-opt %s -ops-specialize=mode=shapes -convert-ops-to-scf | FileCheck %s
// RUN: ops-opt %s -ops-specialize=mode=shapes -convert-ops-to-scf=soa=true | FileCheck %s --check-prefix=SOA

// CHECK-LABEL: func.func @shaped(
// CHECK-SAME: %[[A:arg0]]: memref<6x10xf64, strided<[10, 1], offset: ?>> {llvm.noalias}
// CHECK-SAME: %[[B:arg1]]: memref<6x10x2xf64, strided<[20, 2, 1], offset: ?>> {llvm.noalias}
// SOA-LABEL: func.func @shaped(
// SOA-SAME: %[[A:arg0]]: memref<6x10xf64, strided<[10, 1], offset: ?>> {llvm.noalias}
// SOA-SAME: %[[B:arg1]]: memref<6x10x2xf64, strided<[10, 1, 60], offset: ?>> {llvm.noalias}
ops.chain @shaped(%lo0: index, %hi0: index, %lo1: index, %hi1: index, %a: !ops.dat<f64, 2, 1>, %b: !ops.dat<f64, 2, 2>)
    attributes {ops.range_values = [array<i64: 0, 10, 0, 6>],
                ops.shape_values = [array<i64>, array<i64>, array<i64>, array<i64>, array<i64: 10, 6>, array<i64: 10, 6>]} {
  ops.par_loop range(%lo0, %hi0, %lo1, %hi1)
      dats(%a : !ops.dat<f64, 2, 1> read #ops.stencil<[[0, 0]]>, %b : !ops.dat<f64, 2, 2> write #ops.stencil<[[0, 0]]>)
      scalars() {
  ^bb0(%x: !ops.acc<f64, 2, 1>, %y: !ops.acc<f64, 2, 2>):
    %v = ops.read %x[0, 0] : !ops.acc<f64, 2, 1> -> f64
    ops.write %y[0, 0], %v {comp = 1 : i64} : !ops.acc<f64, 2, 2>
  }
  ops.return
}
